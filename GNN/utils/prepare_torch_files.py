import sys
import os
import time
import json
import torch
import traceback
import numpy as np
import pandas as pd
import uproot
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.data import Data
from build_graph import build_velo_graph, compute_edge_attr

# Adjust the path to where you have modules/functions.py
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
INPUT_FILE_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
OUTPUT_DIR      = "/lustre/LHCb/alejandro.rodriguez/torch_data"
TREE_NAME       = "VeloMultiTuple_73eaa531/Clusters"
STATS_FILE      = "stats/global_normalization_stats.json"

VAR_NAMES = [
    'eventNumber', 'x', 'y', 'z', 'n_pix', 'module', 
    'nVtx_per_event', 'nClu_per_event', 'nTrk_per_event',
    'beamspotX', 'beamspotY'
]

CONT_COLS    = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle', 'module_side']
GLOBAL_COLS  = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']

# ==============================================================================
# LOAD STATISTICS GLOBALLY
# ==============================================================================
try:
    with open(STATS_FILE, 'r') as f:
        global_stats = json.load(f)
    print(f"Loaded global statistics from {STATS_FILE}")
    
    MEANS_CONT = np.array([global_stats[c]['mean'] for c in CONT_COLS],   dtype=np.float32)
    STDS_CONT  = np.array([global_stats[c]['std']  for c in CONT_COLS],   dtype=np.float32) + 1e-8
    MEANS_GLOB = np.array([global_stats[c]['mean'] for c in GLOBAL_COLS], dtype=np.float32)
    STDS_GLOB  = np.array([global_stats[c]['std']  for c in GLOBAL_COLS], dtype=np.float32) + 1e-8
except Exception as e:
    print(f"[ERROR] Could not load statistics: {e}")
    sys.exit(1)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def process_event(args):
    """Converts a single event DataFrame into a PyTorch Geometric Data object
    with a pre-built static graph. Edge attributes are NOT stored — they are
    recomputed on-the-fly during training to keep file sizes small (~10× reduction)."""
    event_id, df_event, is_signal = args

    try:
        # Skip events with too few hits (can't build meaningful graph)
        if len(df_event) < 3:
            return None

        # Continuous features (normalized)
        x_cont = torch.tensor(df_event[CONT_COLS].values, dtype=torch.float)
        # Categorical features (module IDs)
        x_cat = torch.tensor(df_event['module'].values, dtype=torch.long)
        # Global event-level attributes
        global_attr = torch.tensor(df_event[GLOBAL_COLS].iloc[0].values, dtype=torch.float).unsqueeze(0)
        # Physical coordinates (not normalized, in mm) — needed for edge_attr computation
        pos = torch.tensor(df_event[['x_raw', 'y_raw', 'z_raw']].values, dtype=torch.float)
        # Label
        y = torch.tensor([is_signal], dtype=torch.float)

        # ── STATIC GRAPH CONSTRUCTION ──────────────────────────────────────
        # Build module-aware graph on CPU (done once, never recomputed)
        edge_index = build_velo_graph(
            pos, x_cat,
            intra_radius=5.0,   # mm — sensor cluster spread
            inter_k=3,           # neighbours toward adjacent modules
            skip_k=1,            # neighbours toward M±2 (high-pT)
            max_inter_dist=15.0  # mm — conservative cut
        )

        # Skip events that produce no edges (isolated hits)
        if edge_index.shape[1] == 0:
            return None

        # NOTE: edge_attr is NOT stored — it is recomputed in the DataLoader
        # from pos + x_cont. This reduces file size from ~7GB to ~600MB per chunk.

        return Data(
            x_cont=x_cont,
            x_cat=x_cat,
            pos=pos,
            edge_index=edge_index.to(torch.int32),  # int32 saves 50% vs int64
            y=y,
            global_attr=global_attr,
            event_id=torch.tensor([event_id], dtype=torch.long),
            num_nodes=x_cont.shape[0]
        )
    except Exception as e:
        print(f"  [WARN] Event {event_id} failed: {e}", flush=True)
        return None

def run_preparation(label, n_workers=24, test_mode=False):
    """Runs the full preparation pipeline for a specific label."""
    
    # 1. Label-specific configuration
    is_signal = 1 if label == "SIGNAL" else 0
    dec_id = "40114060" if label == "SIGNAL" else "30011001" if label == "MUON" else "38000800"
    input_file_name = f"ntuple_{'signal' if is_signal else 'background'}_{dec_id}.root"
    full_path = f"{INPUT_FILE_PATH}{input_file_name}:{TREE_NAME}"
    
    specific_output_dir = os.path.join(OUTPUT_DIR, 'signal' if is_signal else 'background', dec_id)
    
    # Check for existing files and resume from where we left off
    existing_chunks = set()
    if os.path.exists(specific_output_dir):
        for f in os.listdir(specific_output_dir):
            if f.startswith("graphs_") and f.endswith(".pt"):
                try:
                    idx = int(f.replace("graphs_", "").replace(".pt", ""))
                    existing_chunks.add(idx)
                except ValueError:
                    pass
        if existing_chunks:
            print(f"[{label}] Found {len(existing_chunks)} existing chunks, will RESUME (skip existing)")
    
    os.makedirs(specific_output_dir, exist_ok=True)
    
    print(f"\n[{label}] Starting processing: {full_path}")
    print(f"[{label}] Output: {specific_output_dir}")
    
    chunk_counter = 0
    total_events = 0
    total_skipped = 0
    leftover_df = pd.DataFrame()

    # 2. Iterate in chunks
    try:
        for chunk in uproot.iterate(full_path, VAR_NAMES, step_size="100 MB", library="pd"):
            t0 = time.time()
            
            # Center in beamspot
            chunk['y'] = chunk['y'] - chunk['beamspotY']
            chunk['x'] = chunk['x'] - chunk['beamspotX']
            # Physical cuts: Only remove hits very far behind the collision
            chunk = chunk[chunk['z'] >= -150]

            if not leftover_df.empty:
                chunk = pd.concat([leftover_df, chunk], ignore_index=True)
            if chunk.empty: 
                chunk_counter += 1
                continue

            last_event_id = chunk['eventNumber'].iloc[-1]
            is_last_event = (chunk['eventNumber'] == last_event_id)
            leftover_df = chunk[is_last_event].copy()
            df_to_process = chunk[~is_last_event].copy()

            if df_to_process.empty:
                chunk_counter += 1
                continue

            # --- SKIP if chunk already exists (resume mode) ---
            if chunk_counter in existing_chunks:
                n_events_skip = df_to_process.groupby("eventNumber").ngroups
                total_events += n_events_skip
                total_skipped += n_events_skip
                t = time.time() - t0
                print(f"[{label}] Chunk {chunk_counter}: SKIPPED (already exists, ~{n_events_skip} events) - {t:.1f}s")
                chunk_counter += 1
                continue

            # --- 1. FEATURE ENGINEERING ---
            df_to_process['r_T'], df_to_process['eta'], df_to_process['phi'] = collider_system(df_to_process)
            df_to_process['codex_angle'] = compute_codex_angles(df_to_process)
            df_to_process['module_side'] = df_to_process['module'] % 2

            # No aggressive codex_angle cuts here to prevent breaking track topologies.

            # Keep raw coordinates
            df_to_process['x_raw'], df_to_process['y_raw'], df_to_process['z_raw'] = df_to_process['x'], df_to_process['y'], df_to_process['z']

            # --- 2. NORMALIZATION ---
            df_to_process[CONT_COLS] = (df_to_process[CONT_COLS].values - MEANS_CONT) / STDS_CONT
            df_to_process[GLOBAL_COLS] = (df_to_process[GLOBAL_COLS].values - MEANS_GLOB) / STDS_GLOB

            # --- 3. GRAPH GENERATION (sequential — avoids GIL/memory issues) ---
            events = [(eid, df, is_signal) for eid, df in df_to_process.groupby("eventNumber")]
            
            chunk_data_list = []
            n_failed = 0
            for ev_args in events:
                result = process_event(ev_args)
                if result is not None:
                    chunk_data_list.append(result)
                else:
                    n_failed += 1
            
            total_events += len(chunk_data_list)

            # --- 4. SAVE ---
            chunk_filename = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
            if chunk_data_list:
                torch.save(chunk_data_list, chunk_filename)
                t = time.time() - t0
                failed_msg = f" ({n_failed} skipped)" if n_failed > 0 else ""
                print(f"[{label}] Saved chunk {chunk_counter}: {len(chunk_data_list)} events{failed_msg} "
                      f"(Total: {total_events}) - {t:.1f}s", flush=True)
            
            chunk_counter += 1
            if test_mode:
                print(f"[{label}] Test mode: stopping after 1 chunk.")
                break

    except Exception as e:
        print(f"\n[{label}] ERROR at chunk {chunk_counter}: {e}", flush=True)
        traceback.print_exc()
        print(f"[{label}] Processed {total_events} events before error. Data saved so far is valid.")
        return

    # Final leftover
    if not leftover_df.empty and not test_mode:
        try:
            leftover_df['r_T'], leftover_df['eta'], leftover_df['phi'] = collider_system(leftover_df)
            leftover_df['codex_angle'] = compute_codex_angles(leftover_df)
            leftover_df['module_side'] = leftover_df['module'] % 2
            if not leftover_df.empty:
                leftover_df['x_raw'], leftover_df['y_raw'], leftover_df['z_raw'] = leftover_df['x'], leftover_df['y'], leftover_df['z']
                leftover_df[CONT_COLS] = (leftover_df[CONT_COLS].values - MEANS_CONT) / STDS_CONT
                leftover_df[GLOBAL_COLS] = (leftover_df[GLOBAL_COLS].values - MEANS_GLOB) / STDS_GLOB
                last_graph = process_event((leftover_df['eventNumber'].iloc[0], leftover_df, is_signal))
                if last_graph:
                    torch.save([last_graph], os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt"))
                    total_events += 1
                    print(f"[{label}] Saved final leftover event (Total: {total_events})")
        except Exception as e:
            print(f"[{label}] ERROR processing leftover: {e}", flush=True)

    print(f"[{label}] COMPLETED! {total_events} graphs in {specific_output_dir}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PyTorch Geometric files for CODEX-b.")
    parser.add_argument("--label", type=str, choices=["MUON", "KL0", "SIGNAL", "ALL"], default="KL0",
                        help="Data label to process (or 'ALL' for sequential processing).")
    parser.add_argument("--test_mode", action="store_true", help="Process only 1 chunk for testing.")
    args = parser.parse_args()

    if args.label == "ALL":
        # Process labels SEQUENTIALLY to avoid memory issues with multiprocessing
        labels = ["SIGNAL", "MUON", "KL0"]
        print(f"Processing labels sequentially: {labels}")
        for lbl in labels:
            run_preparation(lbl, n_workers=24, test_mode=args.test_mode)
    else:
        run_preparation(args.label, n_workers=24, test_mode=args.test_mode)