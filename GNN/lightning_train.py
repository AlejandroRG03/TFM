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

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
BKG_TYPE = "MUON"
SIGNAL_DEC_IDS = ["40114060"]
BACKGROUND_DEC_IDS = ["30011001" if BKG_TYPE == "MUON" else "38000800"]

OUTPUT_NAME   = f"{BKG_TYPE}_CODEX_GNN"

BATCH_SIZE    = 32   # Reduced from 128 to avoid CUDA OOM (~16k edges/graph × 128 = 2.1M edges/batch)
ACCUM_STEPS   = 4    # Accumulate 4 steps → effective batch size of 128
EPOCHS        = 100
LEARNING_RATE = 5e-4

TRAIN_SPLIT   = 0.8
PATIENCE      = 15
NUM_WORKERS   = 0      # Set to 0 to avoid multiprocessing fork deadlocks (mmap loads in 0.6s, so async is no longer needed)
VAL_WORKERS   = 0
MAX_VAL_PAIRS = 5      # Limit validation pairs (150 batches × 128 = 19200 events)
USE_MULTI_GPU = False

# Progressive data expansion schedule:
#   (max_epoch_exclusive, num_train_pairs)
#   epochs 0-1:  20 pairs  (~40 min/epoch)
#   epochs 2-6:  70 pairs  (~140 min/epoch)
#   epochs 7+:   all pairs (~280 min/epoch)
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


def estimate_total_events(files: List[str], sample: int = 3) -> int:
    """Estimate total number of events averaging sample files."""
    if not files:
        return 0
    n = min(sample, len(files))
    total = 0
    for f in files[:n]:
        d = torch.load(f, weights_only=False, map_location='cpu')
        total += len(d)
    avg = total / n
    return int(avg * len(files))


# ==============================================================================
# CHUNK LOADING HELPERS (supports both legacy & repacked formats)
# ==============================================================================

def _load_repacked_chunk(filepath: str) -> list:
    """
    Load a chunk file in the repacked (collated tensor dict) format.
    Uses mmap=True for near-instant loading — only pages accessed are read.
    Returns a list of PyG Data objects reconstructed from the collated tensors.
    """
    raw = torch.load(filepath, weights_only=True, map_location='cpu', mmap=True)

    # Discover data keys from the saved dict
    data_keys = [k[5:] for k in raw if k.startswith('data.')]
    num_graphs = raw['slices.y'].size(0) - 1

    graphs = []
    for i in range(num_graphs):
        g = Data()
        for key in data_keys:
            s_start = raw[f'slices.{key}'][i].item()
            s_end   = raw[f'slices.{key}'][i + 1].item()
            tensor  = raw[f'data.{key}']
            # edge_index is (2, E), concatenated along dim=1
            if key == 'edge_index':
                g[key] = tensor[:, s_start:s_end]
            else:
                g[key] = tensor[s_start:s_end]
        g.num_nodes = g.x_cont.size(0)
        graphs.append(g)

    return graphs


def _load_legacy_chunk(filepath: str) -> list:
    """Load a chunk file in the legacy format (pickled list of Data objects)."""
    return torch.load(filepath, weights_only=False, map_location='cpu')


def _is_repacked(filepath: str) -> bool:
    """
    Quick check if file is in repacked format by attempting weights_only load.
    Caches result via a sidecar marker file to avoid repeated probing.
    """
    marker = filepath + '.repacked'
    if os.path.exists(marker):
        return True
    return False


def load_chunk(filepath: str) -> list:
    """Load a chunk file, auto-detecting repacked vs legacy format."""
    if _is_repacked(filepath):
        return _load_repacked_chunk(filepath)
    return _load_legacy_chunk(filepath)


# ==============================================================================
# ITERABLE DATASET (with progressive expansion)
# ==============================================================================

class ChunkIterableDataset(IterableDataset):
    """
    Streams chunk-pairs on the fly. Supports progressive data expansion
    via expansion_schedule and set_epoch().

    current_epoch is a plain int — updated by ProgressiveExpansionCallback.
    DataLoader respawns workers each epoch (persistent_workers=False),
    so workers inherit the updated int via pickle at spawn time.
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

        # DDP split
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

            per_worker = int(math.ceil(len(active_pairs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            active_pairs = active_pairs[worker_id * per_worker : (worker_id + 1) * per_worker]

            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if worker_id == 0:
                print(f"[Epoch {self.current_epoch}][Worker {worker_id}] "
                      f"RAM: {mem_mb:.1f} MB | Active pairs: {n_active}/{len(self.file_pairs)}")

        if not self.is_validation:
            random.shuffle(active_pairs)

        for sig_file, bkg_file in active_pairs:
            # ── Skip corrupt/tiny files (< 100 MB) ──
            try:
                if os.path.getsize(sig_file) < 1e8 or os.path.getsize(bkg_file) < 1e8:
                    continue
            except OSError:
                continue

            # ── Load sig + bkg ──
            try:
                t0 = time.monotonic()
                sig_data = load_chunk(sig_file)
                bkg_data = load_chunk(bkg_file)
                dt = time.monotonic() - t0
                if worker_info and worker_info.id == 0:
                    print(f"[Worker 0] Loaded pair in {dt:.1f}s "
                          f"({len(sig_data)}+{len(bkg_data)} graphs)")
            except Exception as e:
                print(f"[WARN] Failed to load pair: {e}")
                continue

            combined = sig_data + bkg_data
            del sig_data, bkg_data

            if not combined:
                continue

            # ── Ensure x_cat is LongTensor (required by Embedding) ──
            # ── Ensure num_nodes is set ──
            for data in combined:
                if data.x_cat.dtype != torch.long:
                    data.x_cat = data.x_cat.long()
                if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.dtype != torch.long:
                    data.edge_index = data.edge_index.long()
                if not hasattr(data, 'num_nodes') or data.num_nodes is None:
                    data.num_nodes = data.x_cont.size(0)

            # ── BATCHED edge_attr computation ONLY if missing ──
            if not hasattr(combined[0], 'edge_attr') or combined[0].edge_attr is None:
                from utils.build_graph import build_velo_graph, compute_batched_edge_attr
                pos_list = []
                xc_list = []
                ei_list = []
                for data in combined:
                    if not hasattr(data, 'edge_index') or data.edge_index is None:
                        data.edge_index = build_velo_graph(data.pos, data.x_cat)
                    data.edge_index = data.edge_index.to(torch.long)
                    pos_list.append(data.pos)
                    xc_list.append(data.x_cont)
                    ei_list.append(data.edge_index)
                ea_list = compute_batched_edge_attr(pos_list, xc_list, ei_list)
                for data, ea in zip(combined, ea_list):
                    data.edge_attr = ea
            # NOTE: When edge_attr already exists (normal case), we skip
            # the expensive per-graph loop entirely — data is ready to use.

            # Interleave and yield
            indices = list(range(len(combined)))
            random.shuffle(indices)
            for idx in indices:
                yield combined[idx]

            del combined
            gc.collect()


# ==============================================================================
# CALLBACK: progressive expansion
# ==============================================================================

class ProgressiveExpansionCallback(pl.Callback):
    """
    Updates train_dataset.current_epoch at the START of each epoch.
    Workers are respawned at that point (persistent_workers=False),
    so they pickle the dataset with the already-updated epoch number.
    """
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

    # Limit validation pairs to avoid loading unused data
    if len(val_pairs) > MAX_VAL_PAIRS:
        val_pairs = val_pairs[:MAX_VAL_PAIRS]

    print(f"--> Train: {len(train_pairs)} chunk pairs ({len(sig_train)} sig, {len(bkg_train)} bkg)")
    print(f"--> Val:   {len(val_pairs)} chunk pairs (limited to {MAX_VAL_PAIRS})")

    pos_weight_val = 1.0
    print(f"--> Using pos_weight = {pos_weight_val} (chunks are already balanced 1:1)")

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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=None,
        worker_init_fn=lambda _: torch.set_num_threads(1),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=VAL_WORKERS,
        persistent_workers=False,  # FIX: was True, caused memory leak
        pin_memory=True,
        prefetch_factor=None,
        worker_init_fn=lambda _: torch.set_num_threads(1),
    )

    # 4. Auto-detect feature dimension from first chunk pair
    print("--> Detecting feature dimension...")
    _first_sig = load_chunk(train_pairs[0][0])
    detected_n_feats = _first_sig[0].x_cont.shape[-1]
    print(f"    x_cont has {detected_n_feats} features.")
    del _first_sig
    gc.collect()

    model = CODEXLightning(
        pos_weight_val=pos_weight_val,
        learning_rate=LEARNING_RATE,
        model_kwargs={
            "n_cont_features": detected_n_feats,
            "hidden_channels": 128,
            "num_layers": 5,
            "edge_hidden": 96,
            "embedding_dim": 24,
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
    devices_config = [0] if torch.cuda.is_available() else 1

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices_config,
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=ACCUM_STEPS,  # Effective batch = BATCH_SIZE * ACCUM_STEPS = 128
        limit_val_batches=150,
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