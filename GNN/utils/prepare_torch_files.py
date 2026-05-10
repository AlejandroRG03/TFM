import sys
import os
import time
import json
import torch
import numpy as np
import pandas as pd
import uproot
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch_geometric.data import Data

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
    """Converts a single event DataFrame into a PyTorch Geometric Data object."""
    event_id, df_event, is_signal = args
    # No filtering by length anymore, we want all point clouds

    # Continuous features (normalized)
    x_cont = torch.tensor(df_event[CONT_COLS].values, dtype=torch.float)
    # Categorical features (module IDs)
    x_cat = torch.tensor(df_event['module'].values, dtype=torch.long)
    # Global event-level attributes
    global_attr = torch.tensor(df_event[GLOBAL_COLS].iloc[0].values, dtype=torch.float).unsqueeze(0)
    # Physical coordinates for graph construction (not normalized)
    pos = torch.tensor(df_event[['x_raw', 'y_raw', 'z_raw']].values, dtype=torch.float)
    # Label
    y = torch.tensor([is_signal], dtype=torch.float)

    return Data(
        x_cont=x_cont, 
        x_cat=x_cat,
        pos=pos,
        y=y,
        global_attr=global_attr,
        event_id=torch.tensor([event_id], dtype=torch.long),
        num_nodes = x_cont.shape[0]
    )

def run_preparation(label, n_workers=24, test_mode=False):
    """Runs the full preparation pipeline for a specific label."""
    
    # 1. Label-specific configuration
    is_signal = 1 if label == "SIGNAL" else 0
    dec_id = "40114060" if label == "SIGNAL" else "30011001" if label == "MUON" else "38000800"
    input_file_name = f"ntuple_{'signal' if is_signal else 'background'}_{dec_id}.root"
    full_path = f"{INPUT_FILE_PATH}{input_file_name}:{TREE_NAME}"
    
    specific_output_dir = os.path.join(OUTPUT_DIR, 'signal' if is_signal else 'background', dec_id)
    
    if os.path.exists(specific_output_dir):
        import shutil
        print(f"[{label}] Cleaning up residual data in {specific_output_dir}...")
        shutil.rmtree(specific_output_dir)
        
    os.makedirs(specific_output_dir, exist_ok=True)
    
    print(f"\n[{label}] Starting processing: {full_path}")
    print(f"[{label}] Output: {specific_output_dir}")
    
    chunk_counter = 0
    total_events = 0
    leftover_df = pd.DataFrame()

    # 2. Iterate in chunks
    for chunk in uproot.iterate(full_path, VAR_NAMES, step_size="100 MB", library="pd"):
        t0 = time.time()
        
        # Center in beamspot
        chunk['y'] = chunk['y'] - chunk['beamspotY']
        chunk['x'] = chunk['x'] - chunk['beamspotX']
        # Physical cuts: Only remove hits very far behind the collision
        chunk = chunk[chunk['z'] >= -150]

        if not leftover_df.empty:
            chunk = pd.concat([leftover_df, chunk], ignore_index=True)
        if chunk.empty: continue

        last_event_id = chunk['eventNumber'].iloc[-1]
        is_last_event = (chunk['eventNumber'] == last_event_id)
        leftover_df = chunk[is_last_event].copy()
        df_to_process = chunk[~is_last_event].copy()

        if df_to_process.empty: continue

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

        # --- 3. PARALLEL GRAPH GENERATION ---
        events = [(eid, df, is_signal) for eid, df in df_to_process.groupby("eventNumber")]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(process_event, events))

        chunk_data_list = [g for g in results if g is not None]
        total_events += len(chunk_data_list)

        # --- 4. SAVE ---
        chunk_filename = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
        if chunk_data_list:
            torch.save(chunk_data_list, chunk_filename)
            t = time.time() - t0
            print(f"[{label}] Saved chunk {chunk_counter}: {len(chunk_data_list)} events (Total: {total_events}) - {t:.1f}s")
        
        chunk_counter += 1
        if test_mode:
            print(f"[{label}] Test mode: stopping after 1 chunk.")
            break

    # Final leftover
    if not leftover_df.empty and not test_mode:
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

    print(f"[{label}] COMPLETED! {total_events} graphs in {specific_output_dir}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PyTorch Geometric files for CODEX-b.")
    parser.add_argument("--label", type=str, choices=["MUON", "KL0", "SIGNAL", "ALL"], default="KL0",
                        help="Data label to process (or 'ALL' for multiprocessing).")
    parser.add_argument("--test_mode", action="store_true", help="Process only 1 chunk for testing.")
    args = parser.parse_args()

    if args.label == "ALL":
        labels = ["MUON", "KL0", "SIGNAL"]
        print(f"Launching MULTIPROCESSING for labels: {labels}")
        # Use 10 threads per process to stay within 32-core limit (3 * 10 = 30)
        with ProcessPoolExecutor(max_workers=3) as executor:
            executor.map(run_preparation, labels, [10]*3, [args.test_mode]*3)
    else:
        run_preparation(args.label, n_workers=24, test_mode=args.test_mode)