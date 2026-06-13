import sys
import os
import time
import json
import torch
import traceback
import tempfile
import numpy as np
import pandas as pd
import uproot
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, get_context
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from build_graph import build_velo_graph, compute_edge_attr

sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *

INPUT_FILE_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
OUTPUT_DIR      = "/lustre/LHCb/alejandro.rodriguez/torch_data"   # default, overridable via --output-dir
TREE_NAME       = "VeloMultiTuple_73eaa531/Clusters"
STATS_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats/global_normalization_stats.json")

VAR_NAMES = [
    'eventNumber', 'x', 'y', 'z', 'n_pix', 'module',
    'nVtx_per_event', 'nClu_per_event', 'nTrk_per_event',
    'beamspotX', 'beamspotY'
]

CONT_COLS    = ['x', 'y', 'z', 'r_T', 'phi', 'n_pix', 'codex_angle', 'module_side']
GLOBAL_COLS  = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']

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
# WORKER INIT — runs once per subprocess
# ==============================================================================

def _worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


# ==============================================================================
# EVENT PROCESSING (returns filename, not Data, to avoid tensor sharing issues)
# ==============================================================================

def process_event(args):
    """
    Build a single graph from an event DataFrame.

    Returns (event_id, tmp_path_or_None) where tmp_path is the path to
    a saved .pt file on /tmp containing the Data object, or None on failure.
    """
    event_id, df_event, is_signal = args

    try:
        if len(df_event) < 3:
            return (event_id, None)

        x_cont = torch.tensor(df_event[CONT_COLS].values, dtype=torch.float)
        module_ids = torch.tensor(df_event['module'].values, dtype=torch.long)
        global_attr = torch.tensor(df_event[GLOBAL_COLS].iloc[0].values, dtype=torch.float).unsqueeze(0)
        pos = torch.tensor(df_event[['x_raw', 'y_raw', 'z_raw']].values, dtype=torch.float)
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


        # Write to /tmp and return path instead of the Data object itself.
        # This avoids PyTorch's multiprocessing tensor-sharing sockets,
        # which fail on some cluster configurations (LbEnv/CVMFS).
        tmp_path = os.path.join(tempfile.gettempdir(),
                                f"_gnn_{os.getpid()}_{event_id}.pt")
        tmp_path_tmp = tmp_path + ".tmp"
        torch.save(data, tmp_path_tmp)
        os.rename(tmp_path_tmp, tmp_path)
        return (event_id, tmp_path)

    except Exception as e:
        print(f"  [WARN] Event {event_id} failed: {e}", flush=True)
        return (event_id, None)


def run_preparation(label, n_workers=24, test_mode=False, force=False, output_dir=None):
    is_signal = 1 if label == "SIGNAL" else 0
    dec_id = "40114061" if label == "SIGNAL" else "30011001" if label == "MUON" else "38000801"
    input_file_name = f"ntuple_{'signal' if is_signal else 'background'}_{dec_id}.root"
    full_path = f"{INPUT_FILE_PATH}{input_file_name}:{TREE_NAME}"

    base_dir = output_dir if output_dir is not None else OUTPUT_DIR
    specific_output_dir = os.path.join(base_dir, 'signal' if is_signal else 'background', dec_id)
    os.makedirs(specific_output_dir, exist_ok=True)

    existing_chunks = set()
    if os.path.exists(specific_output_dir):
        for f in os.listdir(specific_output_dir):
            if f.startswith("graphs_") and f.endswith(".pt"):
                if os.path.exists(os.path.join(specific_output_dir, f + '.repacked')):
                    try:
                        idx = int(f.replace("graphs_", "").replace(".pt", ""))
                        existing_chunks.add(idx)
                    except ValueError:
                        pass

    if force and existing_chunks:
        print(f"[{label}] --force set, removing {len(existing_chunks)} existing chunks...")
        for f in os.listdir(specific_output_dir):
            if f.startswith("graphs_") and f.endswith(".pt"):
                os.remove(os.path.join(specific_output_dir, f))
                marker = os.path.join(specific_output_dir, f + '.repacked')
                if os.path.exists(marker):
                    os.remove(marker)
        existing_chunks.clear()
        print(f"[{label}] All chunks deleted, starting fresh.")

    if existing_chunks:
        print(f"[{label}] Found {len(existing_chunks)} existing chunks, will RESUME (skip existing)")

    print(f"\n[{label}] Starting processing: {full_path}")
    print(f"[{label}] Output: {specific_output_dir}")

    chunk_counter = 0
    total_events = 0
    total_skipped = 0
    leftover_df = pd.DataFrame()

    # ── Persistent process pool (reused across all chunks in this label) ──
    # Explicit fork context avoids PyTorch tensor-sharing socket errors.
    fork_ctx = get_context('fork')
    pool = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=fork_ctx,
        initializer=_worker_init,
    )

    try:
        for chunk in uproot.iterate(full_path, VAR_NAMES, step_size="100 MB", library="pd"):
            t0 = time.time()

            chunk['y'] = chunk['y'] - chunk['beamspotY']
            chunk['x'] = chunk['x'] - chunk['beamspotX']
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

            if chunk_counter in existing_chunks:
                n_events_skip = df_to_process.groupby("eventNumber").ngroups
                total_events += n_events_skip
                total_skipped += n_events_skip
                t = time.time() - t0
                print(f"[{label}] Chunk {chunk_counter}: SKIPPED (already exists, ~{n_events_skip} events) - {t:.1f}s")
                chunk_counter += 1
                continue

            # --- FEATURE ENGINEERING ---
            df_to_process['r_T'], _, df_to_process['phi'] = collider_system(df_to_process)
            df_to_process['codex_angle'] = compute_codex_angles(df_to_process)
            df_to_process['module_side'] = df_to_process['module'] % 2
            df_to_process['x_raw'], df_to_process['y_raw'], df_to_process['z_raw'] = df_to_process['x'], df_to_process['y'], df_to_process['z']

            # --- NORMALISATION ---
            df_to_process[CONT_COLS] = (df_to_process[CONT_COLS].values - MEANS_CONT) / STDS_CONT
            df_to_process[GLOBAL_COLS] = (df_to_process[GLOBAL_COLS].values - MEANS_GLOB) / STDS_GLOB

            # --- PARALLEL GRAPH GENERATION ---
            events_list = [(eid, df, is_signal) for eid, df in df_to_process.groupby("eventNumber")]
            chunk_data_list = []
            n_failed = 0

            if len(events_list) > 0:
                futures = {pool.submit(process_event, args): args[0] for args in events_list}
                for future in as_completed(futures):
                    event_id, tmp_path = future.result()
                    if tmp_path is not None:
                        data = torch.load(tmp_path, weights_only=False, map_location='cpu')
                        os.remove(tmp_path)
                        chunk_data_list.append(data)
                    else:
                        n_failed += 1

            total_events += len(chunk_data_list)

            # --- SAVE AS REPACKED DICT FORMAT ---
            chunk_filename = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
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
    finally:
        pool.shutdown(wait=True)

    # Final leftover (sequential, 1 event — no pool needed)
    if not leftover_df.empty and not test_mode:
        try:
            leftover_df['r_T'], _, leftover_df['phi'] = collider_system(leftover_df)
            leftover_df['codex_angle'] = compute_codex_angles(leftover_df)
            leftover_df['module_side'] = leftover_df['module'] % 2
            if not leftover_df.empty:
                leftover_df['x_raw'], leftover_df['y_raw'], leftover_df['z_raw'] = leftover_df['x'], leftover_df['y'], leftover_df['z']
                leftover_df[CONT_COLS] = (leftover_df[CONT_COLS].values - MEANS_CONT) / STDS_CONT
                leftover_df[GLOBAL_COLS] = (leftover_df[GLOBAL_COLS].values - MEANS_GLOB) / STDS_GLOB
                _, tmp_path = process_event((leftover_df['eventNumber'].iloc[0], leftover_df, is_signal))
                if tmp_path:
                    last_graph = torch.load(tmp_path, weights_only=False, map_location='cpu')
                    os.remove(tmp_path)
                    data_list = [last_graph]
                    if not hasattr(last_graph, 'num_nodes') or last_graph.num_nodes is None:
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

                    leftover_path = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
                    leftover_tmp = leftover_path + ".tmp"
                    torch.save(save_dict, leftover_tmp)
                    os.rename(leftover_tmp, leftover_path)
                    with open(leftover_path + '.repacked', 'w') as f:
                        f.write('1\n')
                    total_events += 1
                    print(f"[{label}] Saved final leftover event (Total: {total_events})")
        except Exception as e:
            print(f"[{label}] ERROR processing leftover: {e}", flush=True)

    print(f"[{label}] COMPLETED! {total_events} graphs in {specific_output_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PyTorch Geometric files for CODEX-b.")
    parser.add_argument("--label", type=str, choices=["MUON", "KL0", "SIGNAL", "ALL"], default=None,
                        help="Data label to process (or 'ALL' for parallel processing).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated labels, e.g. 'SIGNAL,KL0'. Overrides --label.")
    parser.add_argument("--test_mode", action="store_true", help="Process only 1 chunk for testing.")
    parser.add_argument("--force", action="store_true", help="Delete existing chunks and regenerate from scratch.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR}).")
    args = parser.parse_args()

    if args.labels is not None:
        labels = [lbl.strip() for lbl in args.labels.split(",")]
    elif args.label == "ALL":
        labels = ["SIGNAL", "MUON", "KL0"]
    elif args.label is not None:
        labels = [args.label]
    else:
        labels = ["KL0"]

    if len(labels) == 1:
        run_preparation(labels[0], n_workers=24, test_mode=args.test_mode, force=args.force,
                        output_dir=args.output_dir)
    else:
        workers_per_label = max(1, 24 // len(labels))
        print(f"Processing labels IN PARALLEL: {labels} ({workers_per_label} workers each)")
        procs = []
        for lbl in labels:
            p = Process(target=run_preparation,
                        args=(lbl, workers_per_label, args.test_mode, args.force, args.output_dir))
            p.start()
            procs.append((lbl, p))
        for lbl, p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"[{lbl}] Label FAILED (exit code {p.exitcode})", flush=True)
            else:
                print(f"[{lbl}] Label finished successfully.")
