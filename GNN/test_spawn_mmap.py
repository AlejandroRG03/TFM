#!/usr/bin/env python3
"""Test that spawn + mmap loading works without deadlock."""
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import time, os, glob

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"

class TestChunkDataset(IterableDataset):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi else -1
        print(f"[Worker {wid}] Loading {os.path.basename(self.filepath)}...", flush=True)
        t0 = time.monotonic()
        raw = torch.load(self.filepath, weights_only=True, map_location='cpu', mmap=True)
        data_keys = [k[5:] for k in raw if k.startswith('data.')]
        num_graphs = raw['slices.y'].size(0) - 1
        print(f"[Worker {wid}] Mapped {num_graphs} graphs in {time.monotonic()-t0:.2f}s", flush=True)
        for i in range(min(200, num_graphs)):
            g = Data()
            for key in data_keys:
                s = raw[f'slices.{key}']
                s0, s1 = int(s[i]), int(s[i+1])
                t = raw[f'data.{key}']
                if key == 'edge_index':
                    g[key] = t[:, s0:s1].clone().long()
                elif key == 'x_cat':
                    g[key] = t[s0:s1].clone().long()
                else:
                    g[key] = t[s0:s1].clone()
            g.num_nodes = g.x_cont.size(0)
            yield g
        print(f"[Worker {wid}] Done yielding", flush=True)


if __name__ == '__main__':
    files = sorted(glob.glob(f"{DATA_DIR}/signal/40114060/*.pt"))
    repacked = [f for f in files if os.path.exists(f + '.repacked')]
    fp = repacked[0]

    # Simulate main-process feature detection (same as lightning_train.py)
    print(f"[Main] Feature detection from {os.path.basename(fp)}...")
    raw = torch.load(fp, weights_only=True, map_location='cpu', mmap=True)
    print(f"[Main] n_feats = {raw['data.x_cont'].shape[-1]}")
    del raw

    # DataLoader with spawn
    ds = TestChunkDataset(fp)
    ctx = mp.get_context('spawn')
    loader = DataLoader(ds, batch_size=16, num_workers=2,
                        persistent_workers=True, prefetch_factor=2,
                        multiprocessing_context=ctx)

    print("[Main] Iterating DataLoader with spawn context...")
    t0 = time.monotonic()
    count = 0
    for batch in loader:
        count += 1
        if count == 1:
            print(f"[Main] First batch received! x_cont shape={batch.x_cont.shape}")
        if count >= 5:
            break
    dt = time.monotonic() - t0
    print(f"[Main] Got {count} batches in {dt:.1f}s — SUCCESS!")
