#!/usr/bin/env python3
"""
prepare_torch_files_pileup.py

Simulates Run 5 pile-up conditions by mixing 1 simulated event
with N minimum-bias events. Each output graph corresponds to a
composite 'Run 5' event.

Usage:
  python prepare_torch_files_pileup.py --label SIGNAL --minbias <path>
  python prepare_torch_files_pileup.py --label ALL   --minbias <path>
  python prepare_torch_files_pileup.py --run-tests
  python prepare_torch_files_pileup.py --test_mode   --minbias <path>
"""

import sys
import os
import time
import json
import random
import tempfile
import traceback

import numpy as np
import pandas as pd
import uproot
import argparse

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, get_context

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

from build_graph import build_velo_graph, compute_edge_attr

sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import collider_system, compute_codex_angles

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
MINBIAS_FILE    = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/ntuple_minbias_emilio.root"
OUTPUT_DIR      = "/scratch/alejandro.rodriguez/torch_pileup"
N_MINBIAS       = 7
STEP_SIZE       = "10 MB"
TREE_NAME       = "VeloMultiTuple_73eaa531/Clusters"
STATS_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "stats/global_normalization_stats.json")

VAR_NAMES = [
    'eventNumber', 'x', 'y', 'z', 'n_pix', 'module',
    'nVtx_per_event', 'nClu_per_event', 'nTrk_per_event',
    'beamspotX', 'beamspotY'
]

CONT_COLS   = ['x', 'y', 'z', 'r_T', 'phi', 'n_pix', 'codex_angle', 'module_side']
GLOBAL_COLS = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']

GLOBAL_MEANS_CONT = None
GLOBAL_STDS_CONT  = None
GLOBAL_MEANS_GLOB = None
GLOBAL_STDS_GLOB  = None

try:
    with open(STATS_FILE, 'r') as f:
        global_stats = json.load(f)
    print(f"[INFO] Loaded global statistics from {STATS_FILE}")

    GLOBAL_MEANS_CONT = np.array([global_stats[c]['mean'] for c in CONT_COLS],   dtype=np.float32)
    GLOBAL_STDS_CONT  = np.array([global_stats[c]['std']  for c in CONT_COLS],   dtype=np.float32) + 1e-8
    GLOBAL_MEANS_GLOB = np.array([global_stats[c]['mean'] for c in GLOBAL_COLS], dtype=np.float32)
    GLOBAL_STDS_GLOB  = np.array([global_stats[c]['std']  for c in GLOBAL_COLS], dtype=np.float32) + 1e-8
except Exception as e:
    print(f"[ERROR] Could not load statistics: {e}")
    sys.exit(1)

# ==============================================================================
# GLOBALS — inherited by forked children (copy-on-write)
# ==============================================================================

_MINBIAS_DICT = None   # {eventNumber: DataFrame} once loaded

# ==============================================================================
# MINBIAS SELECTOR — random sampling without immediate replacement
# ==============================================================================

class MinbiasSelector:
    def __init__(self, all_ids):
        self.all_ids = list(all_ids)
        self._reset()

    def _reset(self):
        self.available = list(self.all_ids)
        random.shuffle(self.available)

    def select(self, k):
        if len(self.available) < k:
            self._reset()
        chosen = self.available[:k]
        self.available = self.available[k:]
        return chosen

# ==============================================================================
# MINBIAS PRELOAD
# ==============================================================================

def preload_minbias(path):
    """
    Read all minbias events, apply beamspot centering and z-cut,
    and return a dict {eventNumber: centered_DataFrame}.
    """
    print(f"[INFO] Pre-loading minbias events from {path} ...")
    t0 = time.time()

    full = f"{path}:{TREE_NAME}"
    df = uproot.concatenate(full, VAR_NAMES, library="pd")

    # Beamspot centering
    df['x'] = df['x'] - df['beamspotX']
    df['y'] = df['y'] - df['beamspotY']

    # z-cut (VELO acceptance)
    df = df[df['z'] >= -150].copy()

    # Group by eventNumber → dict of DataFrames
    # Drop beamspot columns to save memory (no longer needed after centering)
    df = df.drop(columns=['beamspotX', 'beamspotY'])
    mb_dict = {eid: grp for eid, grp in df.groupby("eventNumber")}

    # Filter out empty events
    mb_dict = {eid: grp for eid, grp in mb_dict.items() if len(grp) > 0}

    t = time.time() - t0
    print(f"[INFO] Loaded {len(mb_dict)} minbias events ({len(df)} hits) in {t:.1f}s")
    return mb_dict

# ==============================================================================
# EVENT MIXING
# ==============================================================================

def mix_events(event_id, signal_df, minbias_dfs):
    """
    Mix 1 signal event with N minbias events into a single composite DataFrame.

    Args:
        event_id:     signal event number (used as the composite event ID).
        signal_df:    DataFrame for the signal event (centered, z-cut applied).
        minbias_dfs:  list of DataFrames for minbias events (centered, z-cut applied).

    Returns:
        composite_df with summed global attributes and consistent eventNumber.
    """
    composite_df = pd.concat([signal_df] + minbias_dfs, ignore_index=True)

    # Sum global attributes across all constituent events
    nVtx_total = signal_df['nVtx_per_event'].iloc[0]
    nTrk_total = signal_df['nTrk_per_event'].iloc[0]
    for mb_df in minbias_dfs:
        nVtx_total += mb_df['nVtx_per_event'].iloc[0]
        nTrk_total += mb_df['nTrk_per_event'].iloc[0]
    nClu_total = len(composite_df)

    composite_df['eventNumber']        = event_id
    composite_df['nVtx_per_event']    = nVtx_total
    composite_df['nClu_per_event']    = nClu_total
    composite_df['nTrk_per_event']    = nTrk_total

    return composite_df

# ==============================================================================
# WORKER INIT
# ==============================================================================

def _worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

# ==============================================================================
# PROCESS EVENT (runs in pool worker)
# ==============================================================================

def process_event(args):
    """Build a single graph from a composite-event DataFrame.

    Returns (event_id, tmp_path_or_None).
    """
    event_id, df_event, is_signal = args
    try:
        if len(df_event) < 3:
            return (event_id, None)

        x_cont     = torch.tensor(df_event[CONT_COLS].values, dtype=torch.float)
        module_ids = torch.tensor(df_event['module'].values, dtype=torch.long)
        global_attr = torch.tensor(
            df_event[GLOBAL_COLS].iloc[0].values, dtype=torch.float
        ).unsqueeze(0)
        pos = torch.tensor(
            df_event[['x_raw', 'y_raw', 'z_raw']].values, dtype=torch.float
        )
        y = torch.tensor([is_signal], dtype=torch.float)

        edge_index = build_velo_graph(
            pos, module_ids,
            intra_radius=5.0, inter_k=3, skip_k=1, max_inter_dist=15.0
        )
        if edge_index.shape[1] == 0:
            return (event_id, None)

        edge_attr = compute_edge_attr(pos, x_cont, edge_index)

        data = Data(
            x_cont=x_cont, pos=pos,
            edge_index=edge_index.to(torch.int32),
            edge_attr=edge_attr,
            y=y, global_attr=global_attr,
            event_id=torch.tensor([event_id], dtype=torch.long),
            num_nodes=x_cont.shape[0]
        )

        tmp_path = os.path.join(
            tempfile.gettempdir(), f"_gnn_{os.getpid()}_{event_id}.pt"
        )
        tmp_path_tmp = tmp_path + ".tmp"
        torch.save(data, tmp_path_tmp)
        os.rename(tmp_path_tmp, tmp_path)
        return (event_id, tmp_path)

    except Exception as e:
        print(f"  [WARN] Event {event_id} failed: {e}", flush=True)
        return (event_id, None)

# ==============================================================================
# MAIN PREPARATION LOOP
# ==============================================================================

def run_preparation(label, n_workers=24, test_mode=False, force=False,
                    n_minbias=9):
    global _MINBIAS_DICT

    minbias_dict = _MINBIAS_DICT
    if minbias_dict is None:
        print("[ERROR] _MINBIAS_DICT not loaded. Did you run --run-tests "
              "without providing --minbias?")
        return

    minbias_ids = list(minbias_dict.keys())
    if n_minbias > len(minbias_ids):
        print(f"[ERROR] n_minbias ({n_minbias}) exceeds available minbias "
              f"events ({len(minbias_ids)}).")
        return

    # Reseed RNG for this process (important for parallel labels)
    random.seed(os.getpid() + int(time.time() * 1e6) & 0xFFFFFFFF)

    is_signal = 1 if label == "SIGNAL" else 0
    dec_id = ("40114060" if label == "SIGNAL" else
              "30011001" if label == "MUON" else "38000800")
    input_file_name = (f"ntuple_{'signal' if is_signal else 'background'}"
                       f"_{dec_id}.root")
    full_path = f"{INPUT_FILE_PATH}{input_file_name}:{TREE_NAME}"

    specific_output_dir = os.path.join(
        OUTPUT_DIR, 'signal' if is_signal else 'background', dec_id
    )
    os.makedirs(specific_output_dir, exist_ok=True)

    # ── Resume / force ──────────────────────────────────────────────────
    existing_chunks = set()
    if os.path.exists(specific_output_dir):
        for f in os.listdir(specific_output_dir):
            if f.startswith("graphs_") and f.endswith(".pt"):
                if os.path.exists(os.path.join(specific_output_dir,
                                               f + '.repacked')):
                    try:
                        idx = int(f.replace("graphs_", "").replace(".pt", ""))
                        existing_chunks.add(idx)
                    except ValueError:
                        pass

    if force and existing_chunks:
        print(f"[{label}] --force set, removing {len(existing_chunks)} "
              f"existing chunks...")
        for f in os.listdir(specific_output_dir):
            if f.startswith("graphs_") and f.endswith(".pt"):
                os.remove(os.path.join(specific_output_dir, f))
                marker = os.path.join(specific_output_dir, f + '.repacked')
                if os.path.exists(marker):
                    os.remove(marker)
        existing_chunks.clear()
        print(f"[{label}] All chunks deleted, starting fresh.")

    if existing_chunks:
        print(f"[{label}] Found {len(existing_chunks)} existing chunks, "
              f"will RESUME (skip existing)")

    print(f"\n[{label}] Starting processing: {full_path}")
    print(f"[{label}] Output: {specific_output_dir}")

    # ── Main loop ───────────────────────────────────────────────────────
    chunk_counter = 0
    total_events  = 0
    total_skipped = 0
    leftover_df   = pd.DataFrame()

    fork_ctx = get_context('fork')
    pool = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=fork_ctx,
        initializer=_worker_init,
    )

    mb_selector = MinbiasSelector(minbias_ids)

    try:
        for chunk in uproot.iterate(full_path, VAR_NAMES,
                                    step_size=STEP_SIZE, library="pd"):
            t0 = time.time()

            # ── Beamspot centering + z-cut ──────────────────────────
            chunk['x'] = chunk['x'] - chunk['beamspotX']
            chunk['y'] = chunk['y'] - chunk['beamspotY']
            chunk = chunk[chunk['z'] >= -150]

            # ── Prepend leftover ────────────────────────────────────
            if not leftover_df.empty:
                chunk = pd.concat([leftover_df, chunk], ignore_index=True)
            if chunk.empty:
                chunk_counter += 1
                continue

            # ── Split last event as leftover ────────────────────────
            last_event_id   = chunk['eventNumber'].iloc[-1]
            is_last_event   = (chunk['eventNumber'] == last_event_id)
            leftover_df     = chunk[is_last_event].copy()
            df_to_process   = chunk[~is_last_event].copy()

            if df_to_process.empty:
                chunk_counter += 1
                continue

            if chunk_counter in existing_chunks:
                n_events_skip = df_to_process.groupby("eventNumber").ngroups
                total_events  += n_events_skip
                total_skipped += n_events_skip
                t = time.time() - t0
                print(f"[{label}] Chunk {chunk_counter}: SKIPPED "
                      f"(already exists, ~{n_events_skip} compos. events) "
                      f"- {t:.1f}s")
                chunk_counter += 1
                continue

            # ── Build composite events ──────────────────────────────
            composite_dfs = []
            for sig_id, sig_df in df_to_process.groupby("eventNumber"):
                selected_ids  = mb_selector.select(n_minbias)
                selected_dfs  = [minbias_dict[mb_id] for mb_id in selected_ids]
                composite_df  = mix_events(sig_id, sig_df, selected_dfs)
                composite_dfs.append(composite_df)

            if not composite_dfs:
                chunk_counter += 1
                continue

            # ── Feature engineering (vectorised over all composites) ─
            all_composite = pd.concat(composite_dfs, ignore_index=True)

            all_composite['r_T'], _, all_composite['phi'] = \
                collider_system(all_composite)
            all_composite['codex_angle'] = \
                compute_codex_angles(all_composite)
            all_composite['module_side'] = all_composite['module'] % 2
            all_composite['x_raw'] = all_composite['x'].values
            all_composite['y_raw'] = all_composite['y'].values
            all_composite['z_raw'] = all_composite['z'].values

            # ── Normalization ───────────────────────────────────────
            all_composite[CONT_COLS] = (
                all_composite[CONT_COLS].values - GLOBAL_MEANS_CONT
            ) / GLOBAL_STDS_CONT
            all_composite[GLOBAL_COLS] = (
                all_composite[GLOBAL_COLS].values - GLOBAL_MEANS_GLOB
            ) / GLOBAL_STDS_GLOB

            # ── Split back by event for pool submission ─────────────
            events_list = [
                (eid, grp, is_signal)
                for eid, grp in all_composite.groupby("eventNumber")
            ]

            if test_mode:
                print(f"[{label}] TEST MODE — composite event stats "
                      f"(first {min(len(events_list), 5)}):")
                for i, (eid, grp, _) in enumerate(events_list[:5]):
                    print(f"  Event {eid}: {len(grp)} hits", flush=True)

            # ── Parallel graph generation ───────────────────────────
            chunk_data_list = []
            n_failed = 0

            if events_list:
                futures = {
                    pool.submit(process_event, args): args[0]
                    for args in events_list
                }
                for future in as_completed(futures):
                    event_id, tmp_path = future.result()
                    if tmp_path is not None:
                        data = torch.load(tmp_path, weights_only=False,
                                          map_location='cpu')
                        os.remove(tmp_path)
                        chunk_data_list.append(data)
                    else:
                        n_failed += 1

            total_events += len(chunk_data_list)

            # ── Save as repacked dict format ────────────────────────
            chunk_filename = os.path.join(
                specific_output_dir, f"graphs_{chunk_counter}.pt"
            )
            if chunk_data_list:
                for d in chunk_data_list:
                    if not hasattr(d, 'num_nodes') or d.num_nodes is None:
                        d.num_nodes = d.x_cont.size(0)
                collated_data, slices, _ = collate(
                    chunk_data_list[0].__class__,
                    data_list=chunk_data_list,
                    increment=False,
                    add_batch=False
                )
                save_dict = {}
                for key in collated_data.keys():
                    val = collated_data[key]
                    if isinstance(val, torch.Tensor):
                        save_dict[f'data.{key}'] = val.contiguous()
                for key in slices:
                    save_dict[f'slices.{key}'] = slices[key].contiguous()

                chunk_tmp = chunk_filename + ".tmp"
                torch.save(save_dict, chunk_tmp)
                os.rename(chunk_tmp, chunk_filename)
                with open(chunk_filename + '.repacked', 'w') as f:
                    f.write(f'{len(chunk_data_list)}\n')

                t = time.time() - t0
                failed_msg = f" ({n_failed} failed)" if n_failed > 0 else ""
                print(f"[{label}] Saved chunk {chunk_counter}: "
                      f"{len(chunk_data_list)} events{failed_msg} "
                      f"(Total: {total_events}) - {t:.1f}s", flush=True)

            chunk_counter += 1
            if test_mode:
                print(f"[{label}] Test mode: stopping after 1 chunk.")
                break

    except Exception as e:
        print(f"\n[{label}] ERROR at chunk {chunk_counter}: {e}", flush=True)
        traceback.print_exc()
        print(f"[{label}] Processed {total_events} events before error. "
              f"Data saved so far is valid.")
        return
    finally:
        pool.shutdown(wait=True)

    # ── Final leftover (sequential, 1 composite event) ─────────────────
    if not leftover_df.empty and not test_mode:
        try:
            sig_id = int(leftover_df['eventNumber'].iloc[0])
            selected_ids = mb_selector.select(n_minbias)
            selected_dfs = [minbias_dict[mb_id] for mb_id in selected_ids]
            composite_df = mix_events(sig_id, leftover_df, selected_dfs)

            composite_df['r_T'], _, composite_df['phi'] = \
                collider_system(composite_df)
            composite_df['codex_angle'] = \
                compute_codex_angles(composite_df)
            composite_df['module_side'] = composite_df['module'] % 2
            composite_df['x_raw'] = composite_df['x'].values
            composite_df['y_raw'] = composite_df['y'].values
            composite_df['z_raw'] = composite_df['z'].values
            composite_df[CONT_COLS] = (
                composite_df[CONT_COLS].values - GLOBAL_MEANS_CONT
            ) / GLOBAL_STDS_CONT
            composite_df[GLOBAL_COLS] = (
                composite_df[GLOBAL_COLS].values - GLOBAL_MEANS_GLOB
            ) / GLOBAL_STDS_GLOB

            _, tmp_path = process_event(
                (sig_id, composite_df, is_signal)
            )
            if tmp_path:
                last_graph = torch.load(tmp_path, weights_only=False,
                                        map_location='cpu')
                os.remove(tmp_path)
                data_list = [last_graph]
                if not hasattr(last_graph, 'num_nodes') or \
                   last_graph.num_nodes is None:
                    last_graph.num_nodes = last_graph.x_cont.size(0)
                collated_data, slices, _ = collate(
                    data_list[0].__class__,
                    data_list=data_list,
                    increment=False,
                    add_batch=False
                )
                save_dict = {}
                for key in collated_data.keys():
                    val = collated_data[key]
                    if isinstance(val, torch.Tensor):
                        save_dict[f'data.{key}'] = val.contiguous()
                for key in slices:
                    save_dict[f'slices.{key}'] = slices[key].contiguous()

                leftover_path = os.path.join(
                    specific_output_dir, f"graphs_{chunk_counter}.pt"
                )
                leftover_tmp = leftover_path + ".tmp"
                torch.save(save_dict, leftover_tmp)
                os.rename(leftover_tmp, leftover_path)
                with open(leftover_path + '.repacked', 'w') as f:
                    f.write('1\n')
                total_events += 1
                print(f"[{label}] Saved final leftover composite event "
                      f"(Total: {total_events})")
        except Exception as e:
            print(f"[{label}] ERROR processing leftover: {e}", flush=True)

    print(f"[{label}] COMPLETED! {total_events} graphs in "
          f"{specific_output_dir}")

# ==============================================================================
# TESTS
# ==============================================================================

def test_mixing():
    """Unit test for mix_events() with synthetic DataFrames."""
    print("=" * 60)
    print("  Unit test: mix_events()")
    print("=" * 60)

    n_sig = 10
    signal = pd.DataFrame({
        'eventNumber':    [99999] * n_sig,
        'x':              np.random.randn(n_sig) * 10,
        'y':              np.random.randn(n_sig) * 10,
        'z':              np.random.randn(n_sig) * 100 + 50,
        'n_pix':          np.ones(n_sig),
        'module':         np.arange(n_sig) % 10,
        'nVtx_per_event': [5] * n_sig,
        'nClu_per_event': [n_sig] * n_sig,
        'nTrk_per_event': [10] * n_sig,
    })

    n_mb   = 3
    n_mb_evts = 9
    minbias_list = []
    for i in range(n_mb_evts):
        mb = pd.DataFrame({
            'eventNumber':    [100000 + i] * n_mb,
            'x':              np.random.randn(n_mb) * 10,
            'y':              np.random.randn(n_mb) * 10,
            'z':              np.random.randn(n_mb) * 100 + 50,
            'n_pix':          np.ones(n_mb),
            'module':         np.arange(n_mb) % 3,
            'nVtx_per_event': [4] * n_mb,
            'nClu_per_event': [n_mb] * n_mb,
            'nTrk_per_event': [8] * n_mb,
        })
        minbias_list.append(mb)

    expected_hits = n_sig + n_mb_evts * n_mb   # 37
    expected_nVtx = 5 + n_mb_evts * 4          # 41
    expected_nTrk = 10 + n_mb_evts * 8         # 82

    composite = mix_events(99999, signal, minbias_list)

    assert len(composite) == expected_hits, (
        f"Hit count: {len(composite)} != {expected_hits}")
    assert composite['nVtx_per_event'].iloc[0] == expected_nVtx, (
        f"nVtx: {composite['nVtx_per_event'].iloc[0]} != {expected_nVtx}")
    assert composite['nTrk_per_event'].iloc[0] == expected_nTrk, (
        f"nTrk: {composite['nTrk_per_event'].iloc[0]} != {expected_nTrk}")
    assert composite['nClu_per_event'].iloc[0] == expected_hits, (
        f"nClu: {composite['nClu_per_event'].iloc[0]} != {expected_hits}")
    assert np.all(composite['eventNumber'] == 99999), (
        "eventNumber mismatch")
    # Verify all rows have the same nVtx/nTrk/nClu
    assert composite['nVtx_per_event'].nunique() == 1
    assert composite['nTrk_per_event'].nunique() == 1
    assert composite['nClu_per_event'].nunique() == 1

    print(f"  ✅ Hits:     {len(composite)} (expected {expected_hits})")
    print(f"  ✅ nVtx:     {composite['nVtx_per_event'].iloc[0]}")
    print(f"  ✅ nTrk:     {composite['nTrk_per_event'].iloc[0]}")
    print(f"  ✅ nClu:     {composite['nClu_per_event'].iloc[0]}")
    print(f"  ✅ eventNumber: {composite['eventNumber'].iloc[0]}")
    print(f"  ✅ All rows consistent")
    print("=" * 60)
    print("  mix_events() — all tests PASSED!")
    print("=" * 60)


def test_selector():
    """Unit test for MinbiasSelector."""
    print("=" * 60)
    print("  Unit test: MinbiasSelector")
    print("=" * 60)

    ids = list(range(100))
    sel = MinbiasSelector(ids)

    # Single selection: 9 unique
    chosen = sel.select(9)
    assert len(chosen) == 9, f"Expected 9, got {len(chosen)}"
    assert len(set(chosen)) == 9, "Duplicates in single select"

    # 10 more selections: no overlap (90 total, 100 available)
    used = set(chosen)
    for _ in range(10):
        chosen = sel.select(9)
        assert len(set(chosen)) == 9
        assert len(set(chosen) & used) == 0, "Cross-call duplicate"
        used.update(chosen)

    # Now pool should reset (90 used + 9 for the 11th call = 99, still fine)
    # One more should still work
    chosen = sel.select(9)
    assert len(chosen) == 9
    print("  ✅ All selection tests passed")

    # Edge: k=1
    sel2 = MinbiasSelector([1])
    assert sel2.select(1) == [1]
    print("  ✅ Edge case k=1 works")

    # Edge: full reset cycle
    sel3 = MinbiasSelector(list(range(5)))
    assert len(sel3.select(5)) == 5
    assert len(sel3.select(5)) == 5  # reset happened
    print("  ✅ Edge case reset cycle works")

    print("=" * 60)
    print("  MinbiasSelector — all tests PASSED!")
    print("=" * 60)


def run_tests():
    """Run all unit tests (no ROOT files needed)."""
    test_mixing()
    print()
    test_selector()
    print()
    print("✅ All unit tests PASSED!")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare pile-up mixed PyTorch Geometric files for CODEX-b."
    )
    parser.add_argument(
        "--label", type=str,
        choices=["MUON", "KL0", "SIGNAL", "ALL"],
        default="KL0",
        help="Data label to process (or 'ALL' for parallel processing)."
    )
    parser.add_argument(
        "--minbias", type=str, default=MINBIAS_FILE,
        help=f"Path to the minbias ROOT file. (default: {MINBIAS_FILE})"
    )
    parser.add_argument(
        "--n-minbias", type=int, default=N_MINBIAS,
        help="Number of minbias events to mix per signal event (default: 9)."
    )
    parser.add_argument(
        "--test_mode", action="store_true",
        help="Process only 1 chunk for testing."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing chunks and regenerate from scratch."
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="Run unit tests (no ROOT files needed) and exit."
    )
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        sys.exit(0)

    # ── Pre-load minbias (once, before fork) ────────────────────────
    _MINBIAS_DICT = preload_minbias(args.minbias)
    if len(_MINBIAS_DICT) == 0:
        print("[ERROR] No minbias events loaded. Aborting.")
        sys.exit(1)

    # Validate n_minbias
    if args.n_minbias > len(_MINBIAS_DICT):
        print(f"[ERROR] --n-minbias ({args.n_minbias}) exceeds available "
              f"minbias events ({len(_MINBIAS_DICT)}).")
        sys.exit(1)

    # ── Dispatch ────────────────────────────────────────────────────
    if args.label == "ALL":
        labels = ["SIGNAL", "MUON", "KL0"]
        workers_per_label = max(1, 24 // len(labels))
        print(f"[MAIN] Processing labels IN PARALLEL: {labels} "
              f"({workers_per_label} workers each, {args.n_minbias} "
              f"minbias events per composite)")
        procs = []
        for lbl in labels:
            p = Process(
                target=run_preparation,
                args=(lbl, workers_per_label, args.test_mode, args.force,
                      args.n_minbias)
            )
            p.start()
            procs.append((lbl, p))
        for lbl, p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"[{lbl}] Label FAILED (exit code {p.exitcode})",
                      flush=True)
            else:
                print(f"[{lbl}] Label finished successfully.")
    else:
        run_preparation(args.label, n_workers=24,
                        test_mode=args.test_mode, force=args.force,
                        n_minbias=args.n_minbias)
