import os
import glob
import math
import time
import random
import psutil
import gc

import itertools
import warnings
from typing import List, Tuple, Optional

import torch
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", ".*No negative samples in targets.*")
warnings.filterwarnings("ignore", ".*No positive samples in targets.*")

from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from lightning_model import CODEXLightning
import torch.multiprocessing as mp
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
BKG_TYPE = "MUON"
SIGNAL_DEC_IDS = ["40114060"]
BACKGROUND_DEC_IDS = ["30011001" if BKG_TYPE == "MUON" else "38000800"]

OUTPUT_NAME   = f"{BKG_TYPE}_CODEX_GNN"

BATCH_SIZE    = 32     # Avoid CUDA OOM (~16k edges/graph × 128 = 2.1M edges/batch)
ACCUM_STEPS   = 4      # Accumulate 4 steps → effective batch size of 128
EPOCHS        = 100
LEARNING_RATE = 5e-4

TRAIN_SPLIT   = 0.8
PATIENCE      = 10
NUM_WORKERS   = 1       # 1 worker evita contención en OSTs y swap (136 GB RAM sobran)
VAL_WORKERS   = 1
MP_CONTEXT    = 'spawn' # CRITICAL: 'fork' deadlocks with mmap on Lustre
MAX_VAL_PAIRS = 5       # Limit validation pairs
USE_MULTI_GPU = False   # DDP causes NCCL timeouts with slow Lustre I/O (single GPU)

# Progressive data expansion schedule:
#   (max_epoch_exclusive, num_train_pairs)
EXPANSION_SCHEDULE = [
    (2,  20),
    (7,  70),
    (999999, None),  # None means "all"
]

# ==============================================================================
# HELPERS
# ==============================================================================

def get_files(data_dir: str, dec_ids: List[str], data_type: str) -> List[str]:
    files = []
    for dec_id in dec_ids:
        path = os.path.join(data_dir, data_type, dec_id, "*.pt")
        files.extend(sorted(glob.glob(path)))
    return files


def get_paired_files(sig_list: List[str], bkg_list: List[str]) -> List[Tuple[str, str]]:
    if not sig_list or not bkg_list:
        raise ValueError("[ERROR] No signal or background files found.")
    if len(sig_list) > len(bkg_list):
        return list(zip(sig_list, itertools.cycle(bkg_list)))
    return list(zip(itertools.cycle(sig_list), bkg_list))


# ==============================================================================
# CHUNK LOADING (supports both legacy & repacked formats)
# ==============================================================================

def _is_repacked(filepath: str) -> bool:
    """Check if file has been repacked via marker file."""
    return os.path.exists(filepath + '.repacked')


def _detect_n_features(filepath: str) -> int:
    """
    Read just enough of a repacked file to determine n_cont_features.
    Uses mmap=False to avoid contaminating process state before fork.
    Only reads the slices to find the shape — does NOT load all data.
    """
    # For repacked files, we can read just the metadata dict keys
    raw = torch.load(filepath, weights_only=True, map_location='cpu', mmap=True)
    n_feats = raw['data.x_cont'].shape[-1]
    del raw
    gc.collect()
    return n_feats


def _load_repacked_chunk(filepath: str) -> list:
    """
    Load a repacked chunk entirely into RAM (sequential read).

    With Lustre OSTs at 100% capacity, the thousands of tiny page faults
    from mmap + clone are catastrophically slow (~13 min per chunk).
    A single sequential read saturates the available I/O bandwidth and is
    far more reliable on near-full OSTs.
    """
    raw = torch.load(filepath, weights_only=True, map_location='cpu', mmap=False)

    data_keys = [k[5:] for k in raw if k.startswith('data.')]
    num_graphs = raw['slices.y'].size(0) - 1

    graphs = []
    for i in range(num_graphs):
        g = Data()
        for key in data_keys:
            s = raw[f'slices.{key}']
            s0, s1 = int(s[i]), int(s[i + 1])
            tensor = raw[f'data.{key}']
            if key == 'edge_index':
                g[key] = tensor[:, s0:s1].clone().long()
            else:
                g[key] = tensor[s0:s1].clone()
        g.num_nodes = g.x_cont.size(0)
        graphs.append(g)

    return graphs


def _load_legacy_chunk(filepath: str) -> list:
    """Load a chunk file in the legacy format (pickled list of Data objects)."""
    data_list = torch.load(filepath, weights_only=False, map_location='cpu')
    # Ensure dtypes are correct
    for d in data_list:
        if hasattr(d, 'edge_index') and d.edge_index is not None:
            d.edge_index = d.edge_index.long()
        if not hasattr(d, 'num_nodes') or d.num_nodes is None:
            d.num_nodes = d.x_cont.size(0)
    return data_list


def load_chunk(filepath: str) -> list:
    """Load a chunk file, auto-detecting repacked vs legacy format."""
    if _is_repacked(filepath):
        return _load_repacked_chunk(filepath)
    return _load_legacy_chunk(filepath)


# ==============================================================================
# ITERABLE DATASET
# ==============================================================================

class ChunkIterableDataset(IterableDataset):
    """
    Streams chunk-pairs on the fly. Supports progressive data expansion
    via expansion_schedule and set_epoch().
    """
    def __init__(self, file_pairs: List[Tuple[str, str]],
                 expansion_schedule: Optional[List] = None,
                 is_validation: bool = False):
        super().__init__()
        self.file_pairs = file_pairs
        self.expansion_schedule = expansion_schedule
        self.is_validation = is_validation
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _num_active_pairs(self) -> int:
        if self.expansion_schedule is None:
            return len(self.file_pairs)
        for max_epoch, n_pairs in self.expansion_schedule:
            if self.current_epoch < max_epoch:
                return len(self.file_pairs) if n_pairs is None else min(n_pairs, len(self.file_pairs))
        return len(self.file_pairs)

    def __iter__(self):
        import torch.distributed as dist

        # Apply progressive expansion globally BEFORE splitting
        n_active = self._num_active_pairs()
        global_pairs = self.file_pairs[:n_active]

        # DDP split (if running multi-GPU)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            per_rank = int(math.ceil(len(global_pairs) / float(world_size)))
            active_pairs = global_pairs[rank * per_rank : (rank + 1) * per_rank]
        else:
            active_pairs = global_pairs

        # Worker-level split and SEEDING
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = torch.initial_seed() % 2**32
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(active_pairs) / float(n_workers)))
            active_pairs = active_pairs[worker_id * per_worker : (worker_id + 1) * per_worker]

            if worker_id == 0:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / (1024 * 1024)
                print(f"[Epoch {self.current_epoch}][Worker 0] "
                      f"RAM: {mem_mb:.0f} MB | Pairs: {len(active_pairs)} "
                      f"(of {n_active}/{len(self.file_pairs)} total)")

        if not self.is_validation:
            random.shuffle(active_pairs)

        # ═══════════════════════════════════════════════════════════════════
        # TRAINING: double-buffer con hilo loader en background
        #   - Hilo loader: carga chunks secuencialmente y los mete en cola
        #   - Hilo main: extrae de la cola y yield graphs a la GPU
        #   - La GPU nunca espera porque el siguiente par ya está en RAM
        # ═══════════════════════════════════════════════════════════════════
        if not self.is_validation:
            import threading
            import queue as queue_mod

            load_queue = queue_mod.Queue(maxsize=3)
            stop_event = threading.Event()

            def _loader():
                try:
                    for sig_file, bkg_file in active_pairs:
                        if stop_event.is_set():
                            break

                        try:
                            if os.path.getsize(sig_file) < 1e8 or os.path.getsize(bkg_file) < 1e8:
                                continue
                        except OSError:
                            continue

                        try:
                            t0 = time.monotonic()
                            # ── Carga concurrente: sig(OST:1) || bkg(OST:0) ──
                            # torch.load libera el GIL → I/O real en paralelo
                            load_results = {}
                            load_errors = {}
                            def _load_one(key, path):
                                try:
                                    load_results[key] = load_chunk(path)
                                except Exception as e:
                                    load_errors[key] = e
                            t_sig = threading.Thread(
                                target=_load_one, args=('sig', sig_file), daemon=True)
                            t_bkg = threading.Thread(
                                target=_load_one, args=('bkg', bkg_file), daemon=True)
                            t_sig.start()
                            t_bkg.start()
                            t_sig.join()
                            t_bkg.join()
                            if load_errors:
                                raise RuntimeError(f"Concurrent load errors: {load_errors}")
                            sig_data = load_results['sig']
                            bkg_data = load_results['bkg']
                            dt = time.monotonic() - t0
                            if worker_info and worker_info.id == 0:
                                print(f"[W0] Loaded pair in {dt:.1f}s "
                                      f"({len(sig_data)}+{len(bkg_data)} graphs)")
                        except Exception as e:
                            print(f"[WARNING] Failed to load pair "
                                  f"({os.path.basename(sig_file)}, {os.path.basename(bkg_file)}): {e}")
                            continue

                        combined = sig_data + bkg_data
                        del sig_data, bkg_data

                        if not combined:
                            continue

                        if not hasattr(combined[0], 'edge_attr') or combined[0].edge_attr is None:
                            print("[WARNING] Computing edge_attr on the fly! This is slow!")
                            from utils.build_graph import compute_edge_attr
                            for data in combined:
                                data.edge_index = data.edge_index.long()
                                data.edge_attr = compute_edge_attr(data.pos, data.x_cont, data.edge_index)

                        indices = list(range(len(combined)))
                        random.shuffle(indices)

                        # Back-pressure: timeout en put para detectar stop_event
                        while True:
                            try:
                                load_queue.put((combined, indices), timeout=1.0)
                                break
                            except queue_mod.Full:
                                if stop_event.is_set():
                                    return
                finally:
                    load_queue.put(None)

            loader = threading.Thread(target=_loader, daemon=True)
            loader.start()

            try:
                while True:
                    item = load_queue.get()
                    if item is None:
                        break
                    combined, indices = item
                    for idx in indices:
                        yield combined[idx]
                    del combined
            finally:
                stop_event.set()

        # ═══════════════════════════════════════════════════════════════════
        # VALIDATION: bucle síncrono original (pocos pares, rápida)
        # ═══════════════════════════════════════════════════════════════════
        else:
            for sig_file, bkg_file in active_pairs:
                try:
                    if os.path.getsize(sig_file) < 1e8 or os.path.getsize(bkg_file) < 1e8:
                        continue
                except OSError:
                    continue

                try:
                    t0 = time.monotonic()
                    sig_data = load_chunk(sig_file)
                    bkg_data = load_chunk(bkg_file)
                    dt = time.monotonic() - t0
                    if worker_info and worker_info.id == 0:
                        print(f"[W0] Loaded pair in {dt:.1f}s "
                              f"({len(sig_data)}+{len(bkg_data)} graphs)")
                except Exception as e:
                    print(f"[WARNING] Failed to load pair "
                          f"({os.path.basename(sig_file)}, {os.path.basename(bkg_file)}): {e}")
                    continue

                combined = sig_data + bkg_data
                del sig_data, bkg_data

                if not combined:
                    continue

                if not hasattr(combined[0], 'edge_attr') or combined[0].edge_attr is None:
                    print("[WARNING] Computing edge_attr on the fly! This is slow!")
                    from utils.build_graph import compute_edge_attr
                    for data in combined:
                        data.edge_index = data.edge_index.long()
                        data.edge_attr = compute_edge_attr(data.pos, data.x_cont, data.edge_index)

                indices = list(range(len(combined)))
                random.shuffle(indices)
                for idx in indices:
                    yield combined[idx]

                del combined


# ==============================================================================
# CALLBACK: progressive expansion
# ==============================================================================

class ProgressiveExpansionCallback(pl.Callback):
    """Updates train_dataset.current_epoch at the START of each epoch."""
    def __init__(self, train_dataset: ChunkIterableDataset):
        self.train_dataset = train_dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_dataset.set_epoch(trainer.current_epoch)
        n_active = self.train_dataset._num_active_pairs()
        total = len(self.train_dataset.file_pairs)
        print(f"\n[ProgressiveExpansion] Epoch {trainer.current_epoch}: "
              f"using {n_active}/{total} chunk pairs")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train():
    # 1. Discover files
    sig_files = get_files(DATA_DIR, SIGNAL_DEC_IDS, "signal")
    bkg_files = get_files(DATA_DIR, BACKGROUND_DEC_IDS, "background")

    if not sig_files or not bkg_files:
        print("[ERROR] No data files found.")
        return

    print(f"--> Found {len(sig_files)} signal chunks, {len(bkg_files)} background chunks.")

    # 2. Split by FILES (not events) to prevent data leakage
    rng = random.Random(42)
    rng.shuffle(sig_files)
    rng.shuffle(bkg_files)

    n_sig_train = max(1, int(TRAIN_SPLIT * len(sig_files)))
    n_bkg_train = max(1, int(TRAIN_SPLIT * len(bkg_files)))

    sig_train = sig_files[:n_sig_train]
    sig_val   = sig_files[n_sig_train:]
    bkg_train = bkg_files[:n_bkg_train]
    bkg_val   = bkg_files[n_bkg_train:]

    train_pairs = get_paired_files(sig_train, bkg_train)
    val_pairs   = get_paired_files(sig_val,   bkg_val)

    if len(val_pairs) > MAX_VAL_PAIRS:
        val_pairs = val_pairs[:MAX_VAL_PAIRS]

    print(f"--> Train: {len(train_pairs)} chunk pairs ({len(sig_train)} sig, {len(bkg_train)} bkg)")
    print(f"--> Val:   {len(val_pairs)} chunk pairs (limited to {MAX_VAL_PAIRS})")

    # Quick validation: check all files exist and are repacked
    print("--> Validating file pairs...")
    n_bad = 0
    for sig, bkg in train_pairs + val_pairs:
        for fp in (sig, bkg):
            if not os.path.exists(fp):
                print(f"  [WARN] Missing: {fp}")
                n_bad += 1
            elif not _is_repacked(fp) and os.path.getsize(fp) < 1e8:
                print(f"  [WARN] Too small and not repacked: {os.path.basename(fp)}")
                n_bad += 1
    if n_bad == 0:
        print("  All files OK.")
    else:
        print(f"  {n_bad} issues found (will skip at runtime).")

    pos_weight_val = 1.0
    print(f"--> Using pos_weight = {pos_weight_val}")

    # 3. Datasets & DataLoaders
    train_dataset = ChunkIterableDataset(
        train_pairs,
        expansion_schedule=EXPANSION_SCHEDULE,
        is_validation=False,
    )
    val_dataset = ChunkIterableDataset(
        val_pairs,
        expansion_schedule=None,
        is_validation=True,
    )

    # Use 'spawn' context to avoid fork+mmap deadlocks on Lustre.
    # With 'spawn', each worker starts as a fresh Python process (no
    # inherited mmap state), then imports modules and pickles the dataset.
    # persistent_workers=True is essential: spawn startup costs ~10s per
    # worker, but only happens once per training run.
    spawn_ctx = mp.get_context(MP_CONTEXT)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=True,
        prefetch_factor=8 if NUM_WORKERS > 0 else None,
        multiprocessing_context=spawn_ctx if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=VAL_WORKERS,
        persistent_workers=(VAL_WORKERS > 0),
        pin_memory=True,
        prefetch_factor=8 if VAL_WORKERS > 0 else None,
        multiprocessing_context=spawn_ctx if VAL_WORKERS > 0 else None,
    )

    # 4. Auto-detect feature dimension
    #    Read metadata only — do NOT use load_chunk() here because it
    #    would create mmap state in the main process.  With 'spawn'
    #    workers this is less critical, but keeping it lightweight is
    #    still good practice.
    print("--> Detecting feature dimension...")
    if _is_repacked(train_pairs[0][0]):
        detected_n_feats = _detect_n_features(train_pairs[0][0])
    else:
        _first = torch.load(train_pairs[0][0], weights_only=False, map_location='cpu')
        detected_n_feats = _first[0].x_cont.shape[-1]
        del _first
        gc.collect()
    print(f"    x_cont has {detected_n_feats} features.")

    model = CODEXLightning(
        pos_weight_val=pos_weight_val,
        learning_rate=LEARNING_RATE,
        model_kwargs={
            "n_cont_features": detected_n_feats,
            "hidden_channels": 128,
            "num_layers": 5,
            "edge_hidden": 96,
        },
    )

    # 5. Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=PATIENCE,
        verbose=True, mode="min",
    )

    os.makedirs(f"models/{BKG_TYPE}", exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=f"models/{BKG_TYPE}",
        filename=f"{OUTPUT_NAME}_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    expansion_cb = ProgressiveExpansionCallback(train_dataset)

    wandb_logger = WandbLogger(
        project="CODEX-GNN",
        name=OUTPUT_NAME,
        log_model="all",
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if USE_MULTI_GPU else [0],
        strategy=DDPStrategy(timeout=timedelta(minutes=60), find_unused_parameters=False) if USE_MULTI_GPU else "auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=ACCUM_STEPS,
        num_sanity_val_steps=0,
        callbacks=[early_stop, checkpoint, lr_monitor, expansion_cb],
        logger=wandb_logger,
    )

    # 7. Train
    print("--> Starting training with PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    t = end_time - start_time
    print(f"\nTotal time: {int(t / 3600)} h {int(t % 3600 / 60)} min {int(t % 60)} s")