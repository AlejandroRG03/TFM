import os
import glob
import math
import time
import random
import itertools
import warnings
from typing import List, Tuple, Optional

import torch

# Optimizations and warning suppressions
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*No negative samples in targets.*")
warnings.filterwarnings("ignore", ".*No positive samples in targets.*")

from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_model import CODEXLightning

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
BKG_TYPE = "MUON"  # "MUON" or "KL0"
SIGNAL_DEC_IDS = ["40114060"]
BACKGROUND_DEC_IDS = ["30011001" if BKG_TYPE == "MUON" else "38000800"]

OUTPUT_NAME   = f"{BKG_TYPE}_CODEX_GNN"

BATCH_SIZE    = 64   # Fits ~7GB of 24GB VRAM with bf16 — much better GPU utilization
EPOCHS        = 100
LEARNING_RATE = 7e-4  # Slightly higher: linear scaling with larger batch (64 vs 32)

# LIMIT CHUNKS FOR QUICK TESTS
MAX_CHUNKS    = 30   # Set to None to train with all data
TRAIN_SPLIT   = 0.8
PATIENCE      = 10   # More data = slower convergence per epoch, need more patience
NUM_WORKERS   = 2    # More workers can cause more memory issues
USE_MULTI_GPU = False # Single GPU is better: avoids DDP deadlocks + larger batch fills GPU

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_files(data_dir: str, dec_ids: List[str], data_type: str) -> List[str]:
    """Retrieves all .pt files for the given decay IDs and data type."""
    files = []
    for dec_id in dec_ids:
        path = os.path.join(data_dir, data_type, dec_id, "*.pt")
        files.extend(glob.glob(path))
    return files

def get_paired_files(sig_list: List[str], bkg_list: List[str]) -> List[Tuple[str, str]]:
    """Pairs signal and background files, cycling the shorter list if necessary."""
    if not sig_list or not bkg_list:
        raise ValueError("[ERROR] No signal or background files found.")
    
    if len(sig_list) > len(bkg_list):
        return list(zip(sig_list, itertools.cycle(bkg_list)))
    return list(zip(itertools.cycle(sig_list), bkg_list))

# ==============================================================================
# DATASET (ITERABLE)
# ==============================================================================

class ChunkIterableDataset(IterableDataset):
    """
    An IterableDataset that loads PyTorch Geometric data chunks on the fly.
    This prevents RAM exhaustion when dealing with large amounts of graphs.
    
    If the loaded Data objects lack an edge_index (old preprocessing format),
    the module-aware graph is built on-the-fly as a fallback.
    """
    def __init__(self, file_pairs: List[Tuple[str, str]], is_train: bool = True, train_split: float = 0.8, seed: int = 42):
        self.file_pairs = file_pairs
        self.is_train = is_train
        self.train_split = train_split
        self.seed = seed

    def __iter__(self):
        import torch.distributed as dist
        from utils.build_graph import build_velo_graph, compute_edge_attr
        
        # 1. Distribute files among GPUs (DDP) if multiple GPUs are used
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            per_rank = int(math.ceil(len(self.file_pairs) / float(world_size)))
            rank_files = self.file_pairs[rank * per_rank : (rank + 1) * per_rank]
        else:
            rank_files = self.file_pairs

        # 2. Distribute files among DataLoader workers within this GPU
        worker_info = get_worker_info()
        if worker_info is None:
            worker_files = rank_files
        else:
            per_worker = int(math.ceil(len(rank_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_files = rank_files[worker_id * per_worker : (worker_id + 1) * per_worker]

        # Only shuffle the reading order of files during training
        if self.is_train:
            random.shuffle(worker_files)

        for sig_file, bkg_file in worker_files:
            # Load the data chunks into RAM (explicitly to CPU to avoid GPU OOM)
            sig_data = torch.load(sig_file, weights_only=False, map_location='cpu')
            bkg_data = torch.load(bkg_file, weights_only=False, map_location='cpu')
            
            # Perform train/val split ensuring the same indices are used across epochs
            sig_data = self.split_chunk_data(sig_data)
            bkg_data = self.split_chunk_data(bkg_data)
            
            combined_data = sig_data + bkg_data
            
            if self.is_train:
                random.shuffle(combined_data)

            # Yield graphs one by one
            for data in combined_data:
                data.num_nodes = data.x_cont.size(0)
                
                # Build graph from scratch only for truly old data (no edge_index)
                if not hasattr(data, 'edge_index') or data.edge_index is None:
                    data.edge_index = build_velo_graph(data.pos, data.x_cat)
                
                # Ensure edge_index is int64 (may be stored as int32 to save disk)
                data.edge_index = data.edge_index.to(torch.long)
                
                # Always compute edge_attr on-the-fly (<1ms, pure arithmetic)
                # This avoids storing it on disk (~5GB per chunk → 0)
                data.edge_attr = compute_edge_attr(data.pos, data.x_cont, data.edge_index)
                
                yield data

    def split_chunk_data(self, data_list: list) -> list:
        """Splits the chunk deterministically based on the fixed seed."""
        gen = random.Random(self.seed)
        indices = list(range(len(data_list)))
        gen.shuffle(indices)
        
        split_idx = int(self.train_split * len(data_list))
        target_indices = indices[:split_idx] if self.is_train else indices[split_idx:]
        
        return [data_list[i] for i in target_indices]

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train():
    # 1. Prepare file list
    sig_files = get_files(DATA_DIR, SIGNAL_DEC_IDS, "signal")
    bkg_files = get_files(DATA_DIR, BACKGROUND_DEC_IDS, "background")
    
    if not sig_files or not bkg_files:
        print("[ERROR] No data files found.")
        return

    # 2. Limit the number of chunks (for quick testing)
    if MAX_CHUNKS is not None:
        sig_files, bkg_files = sig_files[:MAX_CHUNKS], bkg_files[:MAX_CHUNKS]
        print(f"--> [TEST MODE] Using a maximum of {MAX_CHUNKS} chunks.")

    # Shuffle before pairing
    random.shuffle(sig_files)
    random.shuffle(bkg_files)

    paired_files = get_paired_files(sig_files, bkg_files)
    print(f"--> Dataset: {len(paired_files)} chunk pairs ready to use.")

    # 3. Set up DataLoaders (PyTorch Geometric)
    train_dataset = ChunkIterableDataset(paired_files, is_train=True, train_split=TRAIN_SPLIT)
    val_dataset = ChunkIterableDataset(paired_files, is_train=False, train_split=TRAIN_SPLIT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)

    pos_weight_val = float(len(bkg_files)) / max(1, len(sig_files))
    model = CODEXLightning(pos_weight_val=pos_weight_val, learning_rate=LEARNING_RATE)
    
    # 4.5 Apply torch.compile for massive kernel fusion speedup (PyTorch 2.0+)
    # if hasattr(torch, "compile"):
    #     print("--> Enabling torch.compile()...")
    #     model = torch.compile(model)

    # 5. Set up Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=PATIENCE,
        verbose=True,
        mode="min"
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename=f"{OUTPUT_NAME}_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')


    # 6. Initialize PyTorch Lightning Trainer
    if USE_MULTI_GPU and torch.cuda.is_available():
        devices_config = "auto"
        strategy_config = "ddp"
    else:
        devices_config = [0] if torch.cuda.is_available() else 1
        strategy_config = "auto"

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices_config,
        strategy=strategy_config,
        precision="bf16-mixed",  # BF16: same range as fp32, no overflow risk
        gradient_clip_val=1.0,   # Prevent gradient explosions
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor]
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
