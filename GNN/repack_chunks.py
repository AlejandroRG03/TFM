#!/usr/bin/env python3
"""
Repack chunk .pt files from legacy format (pickled list of PyG Data objects)
to a collated tensor-dict format that supports torch.load(mmap=True).

This dramatically reduces loading time from ~3-4 min per chunk to near-instant.

Usage:
    python repack_chunks.py                      # repack all signal + background
    python repack_chunks.py --dry-run             # just report what would be done
    python repack_chunks.py --signal-only         # repack only signal files
    python repack_chunks.py --file /path/to/f.pt  # repack a single file

The script repacks IN-PLACE: it writes a .tmp file, then atomically replaces
the original. This requires only ~6.5 GB of temporary extra disk space at any
time, making it safe even on a nearly-full Lustre filesystem.

A .repacked marker file is created next to each repacked file so the training
pipeline can detect the format without probing.
"""

import os
import sys
import glob
import time
import argparse
import gc

import torch
from torch_geometric.data.collate import collate


DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
SIGNAL_DEC_IDS = ["40114060"]
BKG_TYPE = "KL0"  # To match lightning_train.py
BACKGROUND_DEC_IDS = ["30011001" if BKG_TYPE == "MUON" else "38000800"]


def is_already_repacked(filepath: str) -> bool:
    """Check if file has already been repacked via marker file."""
    return os.path.exists(filepath + '.repacked')


def repack_one_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Repack a single chunk file in-place.

    Returns a dict with stats: {'n_graphs': int, 'old_size': int, 'new_size': int, 'time': float}
    """
    if is_already_repacked(filepath):
        return {'skipped': True, 'reason': 'already repacked'}

    old_size = os.path.getsize(filepath)
    basename = os.path.basename(filepath)

    if dry_run:
        return {'skipped': False, 'dry_run': True, 'old_size': old_size}

    t0 = time.monotonic()

    # 1. Load legacy format
    print(f"  Loading {basename} ({old_size / 1e9:.2f} GB)...", flush=True)
    data_list = torch.load(filepath, weights_only=False, map_location='cpu')
    n_graphs = len(data_list)
    print(f"  Loaded {n_graphs} graphs. Collating...", flush=True)

    # 2. Ensure all graphs have num_nodes set (required by collate)
    for d in data_list:
        if not hasattr(d, 'num_nodes') or d.num_nodes is None:
            d.num_nodes = d.x_cont.size(0)

    # 3. Collate into a single Data object + slices
    #    increment=False: edge_index stays local (0-based per graph)
    #    add_batch=False: no batch vector added
    collated_data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False
    )

    # 4. Build a flat dict of contiguous tensors (mmap-friendly)
    save_dict = {}
    for key in collated_data.keys():
        val = collated_data[key]
        if isinstance(val, torch.Tensor):
            save_dict[f'data.{key}'] = val.contiguous()
    for key in slices:
        save_dict[f'slices.{key}'] = slices[key].contiguous()

    # Free the original data ASAP
    del data_list, collated_data, slices
    gc.collect()

    # 5. Save to temp file, then atomically replace
    tmp_path = filepath + '.tmp'
    print(f"  Saving repacked file...", flush=True)
    torch.save(save_dict, tmp_path)
    del save_dict
    gc.collect()

    new_size = os.path.getsize(tmp_path)

    # Atomic replace (works on same filesystem)
    os.replace(tmp_path, filepath)

    # Create marker file
    with open(filepath + '.repacked', 'w') as f:
        f.write(f'{n_graphs}\n')

    dt = time.monotonic() - t0
    ratio = new_size / old_size * 100

    print(f"  Done: {n_graphs} graphs | "
          f"{old_size / 1e9:.2f} -> {new_size / 1e9:.2f} GB ({ratio:.1f}%) | "
          f"{dt:.1f}s", flush=True)

    return {
        'skipped': False,
        'n_graphs': n_graphs,
        'old_size': old_size,
        'new_size': new_size,
        'time': dt,
    }


def get_all_files():
    """Get all signal and background chunk files."""
    files = []
    for dec_id in SIGNAL_DEC_IDS:
        path = os.path.join(DATA_DIR, "signal", dec_id, "*.pt")
        files.extend(sorted(glob.glob(path)))
    for dec_id in BACKGROUND_DEC_IDS:
        path = os.path.join(DATA_DIR, "background", dec_id, "*.pt")
        files.extend(sorted(glob.glob(path)))
    return files


def main():
    parser = argparse.ArgumentParser(description="Repack chunk files for mmap loading")
    parser.add_argument('--dry-run', action='store_true', help="Just report, don't repack")
    parser.add_argument('--signal-only', action='store_true', help="Only repack signal files")
    parser.add_argument('--background-only', action='store_true', help="Only repack background files")
    parser.add_argument('--file', type=str, help="Repack a single file")
    parser.add_argument('--limit', type=int, default=None, help="Repack at most N files")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    elif args.signal_only:
        files = []
        for dec_id in SIGNAL_DEC_IDS:
            files.extend(sorted(glob.glob(os.path.join(DATA_DIR, "signal", dec_id, "*.pt"))))
    elif args.background_only:
        files = []
        for dec_id in BACKGROUND_DEC_IDS:
            files.extend(sorted(glob.glob(os.path.join(DATA_DIR, "background", dec_id, "*.pt"))))
    else:
        files = get_all_files()

    # Filter out already repacked
    pending = [f for f in files if not is_already_repacked(f)]
    already_done = len(files) - len(pending)

    if args.limit:
        pending = pending[:args.limit]

    print(f"=== Repack Chunks ===")
    print(f"Total files: {len(files)}")
    print(f"Already repacked: {already_done}")
    print(f"To process: {len(pending)}")
    if args.dry_run:
        print("(DRY RUN — no changes will be made)")
    print()

    total_old = 0
    total_new = 0
    total_graphs = 0

    for i, filepath in enumerate(pending):
        print(f"[{i + 1}/{len(pending)}] {os.path.basename(filepath)}")
        result = repack_one_file(filepath, dry_run=args.dry_run)

        if not result.get('skipped') and not result.get('dry_run'):
            total_old += result['old_size']
            total_new += result['new_size']
            total_graphs += result['n_graphs']

    if total_old > 0:
        print(f"\n=== Summary ===")
        print(f"Total graphs repacked: {total_graphs}")
        print(f"Total size: {total_old / 1e9:.2f} -> {total_new / 1e9:.2f} GB "
              f"({total_new / total_old * 100:.1f}%)")
        saved = total_old - total_new
        if saved > 0:
            print(f"Space saved: {saved / 1e9:.2f} GB")
        elif saved < 0:
            print(f"Space increase: {-saved / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
