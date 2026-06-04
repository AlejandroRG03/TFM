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

# Pinned memory limit (informational only — not a CUDA allocator option).
PINNED_MEMORY_LIMIT = 50 * 10**9  # 50 GB
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
BKG_TYPE = "SEPPARATE_BKG" # check if our net can distinguish between muon and kl0, ignoring signal

DISCRIMINATE_BKG = (BKG_TYPE == "SEPPARATE_BKG")

if DISCRIMINATE_BKG:
    SIGNAL_DEC_IDS     = ["30011001"]   # muons as "signal" (y=1)
    BACKGROUND_DEC_IDS = ["38000800"]   # KL0 as "background" (y=0)
    SIGNAL_DATA_TYPE   = "background"   # both live in background/ on disk
else:
    SIGNAL_DEC_IDS     = ["40114060"]
    BACKGROUND_DEC_IDS = ["30011001" if BKG_TYPE == "MUON" else "38000800"]
    SIGNAL_DATA_TYPE   = "signal"

OUTPUT_NAME   = f"{BKG_TYPE}_CODEX_GNN"

BATCH_SIZE    = 128    # H100: 100GB VRAM permite batches grandes
ACCUM_STEPS   = 1      # effective batch = 128
EPOCHS        = 100
LEARNING_RATE = 5e-4

TRAIN_SPLIT   = 0.8
PATIENCE      = 10
NUM_WORKERS   = 6       # H100: 64 cores
VAL_WORKERS   = 4
MP_CONTEXT    = 'spawn' # CRITICAL: 'fork' deadlocks with mmap on Lustre
MAX_VAL_PAIRS = 5       # Limit validation pairs
USE_MULTI_GPU = False   # DDP causes NCCL timeouts with slow Lustre I/O (single GPU)

# Fixed number of training chunk pairs (None = all available).
MAX_TRAIN_PAIRS = 20  # rapido: ~20 pares para esta prueba

# Per-worker chunk cache limit: 80% of available RAM ÷ NUM_WORKERS.
# Auto-scales: more workers → less cache per worker, preventing OOM.
_AVAILABLE_RAM = psutil.virtual_memory().available
_MAX_CACHE_BYTES: int = int((_AVAILABLE_RAM * 0.80) / NUM_WORKERS)

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

# Per-process LRU cache for loaded chunks.
# With persistent_workers=True, workers keep this cache across epochs,
# avoiding repeated Lustre I/O for chunks that are reused.
# _MAX_CACHE_BYTES is computed in config above: 70% of RAM / NUM_WORKERS.
_CHUNK_CACHE: dict = {}
_CACHE_BYTES: int = 0


def _try_cache(filepath: str, data: list) -> None:
    """Insert data into cache if under the total size limit."""
    global _CACHE_BYTES
    try:
        fsize = os.path.getsize(filepath)
        if fsize > 0 and _CACHE_BYTES + fsize < _MAX_CACHE_BYTES:
            _CHUNK_CACHE[filepath] = data
            _CACHE_BYTES += fsize
    except OSError:
        pass


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
    """Load a chunk file, using per-process cache (100 GB limit)."""
    cached = _CHUNK_CACHE.get(filepath)
    if cached is not None:
        return cached
    if _is_repacked(filepath):
        data = _load_repacked_chunk(filepath)
    else:
        data = _load_legacy_chunk(filepath)
    _try_cache(filepath, data)
    return data


# ==============================================================================
# ITERABLE DATASET
# ==============================================================================

class ChunkIterableDataset(IterableDataset):
    """
    Streams chunk-pairs on the fly. Uses a fixed set of file pairs
    (no progressive expansion).
    """
    def __init__(self, file_pairs: List[Tuple[str, str]],
                 is_validation: bool = False,
                 relabel_signal: bool = False):
        super().__init__()
        self.file_pairs = file_pairs
        self.is_validation = is_validation
        self.relabel_signal = relabel_signal

    def __iter__(self):
        import torch.distributed as dist

        global_pairs = self.file_pairs

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
                print(f"[Worker 0] "
                      f"RAM: {mem_mb:.0f} MB | Pairs: {len(active_pairs)} "
                      f"(of {len(self.file_pairs)} total)")

        if not self.is_validation:
            random.shuffle(active_pairs)

        if not self.is_validation:
            import threading
            import queue as queue_mod

            import ctypes
            _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
            load_queue = queue_mod.Queue(maxsize=2)
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
                            if self.relabel_signal:
                                for d in sig_data:
                                    d.y = torch.ones_like(d.y)
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
                    _malloc_trim(0)
            finally:
                stop_event.set()

        # VALIDATION
        else:
            import ctypes
            _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
            all_graphs = []
            load_times = []
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
                    if self.relabel_signal:
                        for d in sig_data:
                            d.y = torch.ones_like(d.y)
                    dt = time.monotonic() - t0
                    load_times.append(dt)
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

                all_graphs.extend(combined)
                del combined
                _malloc_trim(0)

            if worker_info and worker_info.id == 0 and load_times:
                avg = sum(load_times) / len(load_times)
                print(f"[W0] Validation: loaded {len(all_graphs)} graphs "
                      f"from {len(load_times)} pairs ({avg:.1f}s avg load)")

            indices = list(range(len(all_graphs)))
            random.shuffle(indices)
            for idx in indices:
                yield all_graphs[idx]
            del all_graphs, indices
            _malloc_trim(0)


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train():
    # 1. Discover files
    sig_files = get_files(DATA_DIR, SIGNAL_DEC_IDS, SIGNAL_DATA_TYPE)
    bkg_files = get_files(DATA_DIR, BACKGROUND_DEC_IDS, "background")

    if not sig_files or not bkg_files:
        print("[ERROR] No data files found.")
        return

    print(f"--> Found {len(sig_files)} signal chunks, {len(bkg_files)} background chunks.")

    # 2. Hold out last 5 chunks of each species as test set (model never sees these)
    N_TEST = 5
    sig_test = sig_files[-N_TEST:]
    sig_files = sig_files[:-N_TEST]
    bkg_test = bkg_files[-N_TEST:]
    bkg_files = bkg_files[:-N_TEST]
    print(f"--> Test set: {len(sig_test)} sig + {len(bkg_test)} bkg chunks (reserved, never trained/validated)")

    # 3. Split remaining by FILES (not events) to prevent data leakage
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

    ram = psutil.virtual_memory()
    print(f"--> RAM: {ram.available / 1e9:.0f} GB available, "
          f"cache limit: {_MAX_CACHE_BYTES / 1e9:.0f} GB/worker, "
          f"pinned limit: {PINNED_MEMORY_LIMIT / 1e9:.0f} GB")

    # 3. Datasets & DataLoaders
    n_train = len(train_pairs)
    if MAX_TRAIN_PAIRS is not None:
        train_pairs = train_pairs[:MAX_TRAIN_PAIRS]
        print(f"   Using {len(train_pairs)}/{n_train} train pairs (limited by MAX_TRAIN_PAIRS={MAX_TRAIN_PAIRS})")

    train_dataset = ChunkIterableDataset(
        train_pairs,
        is_validation=False,
        relabel_signal=DISCRIMINATE_BKG,
    )
    val_dataset = ChunkIterableDataset(
        val_pairs,
        is_validation=True,
        relabel_signal=DISCRIMINATE_BKG,
    )

    # Use 'spawn' context to avoid fork+mmap deadlocks on Lustre.
    # persistent_workers=False to avoid deadlocks between spawn workers
    # and the ChunkIterableDataset's internal daemon threads.
    # Workers are re-created each epoch (LRU cache lost, but no deadlock).
    spawn_ctx = mp.get_context(MP_CONTEXT)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=32 if NUM_WORKERS > 0 else None,
        multiprocessing_context=spawn_ctx if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=VAL_WORKERS,
        persistent_workers=False,
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
            "edge_hidden": 128,
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
        callbacks=[early_stop, checkpoint, lr_monitor],
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