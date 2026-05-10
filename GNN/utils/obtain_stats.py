import sys
# Adjust the path to where you have modules/functions.py
sys.path.append("/home3/alejandro.rodriguez/python_modules")

from functions import *
import uproot
import json
import numpy as np
import dask
import dask_awkward as dak
import awkward as ak
import os
import time

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
INPUT_FILE_NAME_BACKGROUND = "ntuple_background_30011001.root"
INPUT_FILE_NAME_SIGNAL = "ntuple_signal_40114060.root"
OUTPUT_FILE = "stats/global_normalization_stats.json"

INPUT_FILE_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"

# Variables to extract from ROOT
VAR_NAMES = [
    'eventNumber', 'x', 'y', 'z', 'n_pix', 'module', 
    'nVtx_per_event', 'nClu_per_event', 'nTrk_per_event',
    'beamspotX', 'beamspotY' # for centering the coordinates in the beamspot
]
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

FULL_PATH_SIGNAL = os.path.join(INPUT_FILE_PATH, INPUT_FILE_NAME_SIGNAL) + f":{TREE_NAME}"
FULL_PATH_BACKGROUND = os.path.join(INPUT_FILE_PATH, INPUT_FILE_NAME_BACKGROUND) + f":{TREE_NAME}"

# CODEX-B Center for angle calculation
CODEX_X, CODEX_Y, CODEX_Z = 23725.0, 0.0, 12650.0

# ==============================================================================
# FAST DASK COMPUTATION
# ==============================================================================

def compute_fast_statistics():
    print(f"Initializing Dask computation graph for Signal and Background...")
    start_time = time.time()
    
    # --- 1. Lazy Load with Dask Awkward ---
    # Use step_size of 1M to balance overhead and RAM usage per worker
    events = uproot.dask([FULL_PATH_SIGNAL, FULL_PATH_BACKGROUND], step_size=1_000_000)
    events = events[VAR_NAMES]
    
    # Center x and y in the beamspot (assuming beamspotX and beamspotY are available in the data)
    events['x'] = events['x'] - events['beamspotX']
    events['y'] = events['y'] - events['beamspotY']

    # Filtramos los hits (solo z >= -150) para que las estadísticas coincidan con los datos de entrenamiento
    events = events[events.z >= -150]

    # --- 2. Lazy Feature Engineering ---
    r_T = np.sqrt(events.x**2 + events.y**2)
    phi = np.arctan2(events.y, events.x)
    
    # Robust eta calculation using min/max instead of clip for ufunc compatibility
    theta = np.arctan2(r_T, events.z)
    theta_half = theta / 2.0
    theta_clipped = np.minimum(np.maximum(theta_half, 1e-7), np.pi/2 - 1e-7)
    eta = -np.log(np.tan(theta_clipped))
    
    # Vectorized Codex Angle calculation (dot product)
    norm_hit = np.sqrt(events.x**2 + events.y**2 + events.z**2)
    norm_codex = np.sqrt(CODEX_X**2 + CODEX_Y**2 + CODEX_Z**2)
    dot_product = (events.x * CODEX_X + events.y * CODEX_Y + events.z * CODEX_Z)
    
    # Use min/max instead of clip for compatibility
    cos_angle_raw = dot_product / (norm_hit * norm_codex)
    cos_angle = np.minimum(np.maximum(cos_angle_raw, -1.0), 1.0)
    codex_angle = np.arccos(cos_angle)
    
    # Add engineered features to the virtual array
    events["r_T"] = r_T
    events["phi"] = phi
    events["eta"] = eta
    events["codex_angle"] = codex_angle 

    events["module_side"] = events["module"] % 2

    # --- 3. Column Definitions ---
    hit_cols = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle', 'module_side']
    event_cols = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']
    all_cols = hit_cols + event_cols
    
    stats_dict = {}
    
    # --- 4. Graph Construction (Avoiding complex unimplemented reductions) ---
    print("Building computational graph (sum, sum_sq, count, min, max)...")
    for col in all_cols:
        stats_dict[f"{col}_sum"] = dak.sum(events[col])
        stats_dict[f"{col}_sum_sq"] = dak.sum(events[col]**2)
        stats_dict[f"{col}_count"] = dak.count(events[col])
        stats_dict[f"{col}_min"] = dak.min(events[col])
        stats_dict[f"{col}_max"] = dak.max(events[col])

    # --- 5. Parallel Execution ---
    print("Executing parallel computation across all CPU cores... Please wait.")
    computed_stats = dask.compute(stats_dict)[0]

    # --- 6. Local Reconstruction of Mean and Std Dev ---
    print("Post-processing results locally...")
    final_stats = {}
    for col in all_cols:
        c = computed_stats[f"{col}_count"]
        
        if c > 0:
            mean = computed_stats[f"{col}_sum"] / c
            # Var = E[X^2] - (E[X])^2
            var = max((computed_stats[f"{col}_sum_sq"] / c) - (mean**2), 0)
            std = np.sqrt(var) if var > 0 else 0.0
        else:
            mean, std = 0.0, 0.0

        final_stats[col] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(computed_stats[f"{col}_min"]),
            'max': float(computed_stats[f"{col}_max"])
        }
        

    final_stats['metadata'] = {
        'input_files': [INPUT_FILE_NAME_SIGNAL, INPUT_FILE_NAME_BACKGROUND],
        'computed_at': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'n_hits_total': int(computed_stats['x_count'])
    }

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")
    return final_stats

if __name__ == "__main__":
    # Ensure the stats directory exists
    os.makedirs('stats', exist_ok=True)
    
    try:
        stats = compute_fast_statistics()
        print("\nStatistics computed successfully!")
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics successfully saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\n[ERROR] An exception occurred during execution:")
        print(e)