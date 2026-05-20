#!/usr/bin/env python3
"""Quick test: time a single repacked file load via mmap vs shm copy."""
import torch
import time
import glob
import os
import shutil
import tempfile

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
files = sorted(glob.glob(f"{DATA_DIR}/signal/40114060/*.pt"))
# pick one that is repacked
repacked = [f for f in files if os.path.exists(f + '.repacked')]
if not repacked:
    print("No repacked files found")
    exit(1)

fp = repacked[0]
print(f"Testing: {fp} ({os.path.getsize(fp)/1e9:.2f} GB)")

# Method 1: mmap=True directly from Lustre
print("\n--- Method 1: mmap=True directly from Lustre ---")
t0 = time.monotonic()
raw = torch.load(fp, weights_only=True, map_location='cpu', mmap=True)
t1 = time.monotonic()
print(f"  torch.load(mmap=True):  {t1-t0:.3f}s")

# Access some data to force actual page-in
t2 = time.monotonic()
# Force read of edge_index
ei = raw['data.edge_index']
_ = ei[0, :100].clone()
t3 = time.monotonic()
print(f"  First page-in (edge_index slice): {t3-t2:.3f}s")

del raw

# Method 2: copy to /dev/shm then mmap=False
print("\n--- Method 2: copy to /dev/shm then load ---")
fd, shm_path = tempfile.mkstemp(dir='/dev/shm', prefix='test_', suffix='.pt')
os.close(fd)
t4 = time.monotonic()
shutil.copyfile(fp, shm_path)
t5 = time.monotonic()
print(f"  shutil.copyfile to /dev/shm: {t5-t4:.3f}s")
raw2 = torch.load(shm_path, weights_only=True, map_location='cpu', mmap=False)
t6 = time.monotonic()
print(f"  torch.load(mmap=False) from shm: {t6-t5:.3f}s")
os.remove(shm_path)
del raw2

print(f"\nConclusion: mmap direct is {t1-t0:.3f}s vs shm copy+load {(t6-t4):.3f}s total")
