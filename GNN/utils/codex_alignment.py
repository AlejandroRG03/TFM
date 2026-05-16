"""
CODEX-b Alignment Validation Script.
This script identifies track segments (pairs of hits in adjacent modules) 
that are confirmed by a third hit and evaluates their alignment with 
the CODEX-b detector axis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *
set_tfm_style()

# Optional: use a KD-tree for faster nearest-neighbour queries
try:
    from scipy.spatial import cKDTree
    HAVE_CKD = True
except ImportError:
    HAVE_CKD = False

# ==========================================
# 1. GEOMETRIC PARAMETERS AND FILTERS
# ==========================================
EXTRAP_TOL = 0.5  # mm (Tolerance for hit matching)
USE_XY = True

CODEX_X, CODEX_Y, CODEX_Z = 23725, 0, 12650
CODEX_L = 10_000
CODEX_Z_FRONT = CODEX_Z - CODEX_L / 2

CODEX_CENTER = np.array([CODEX_X, CODEX_Y, CODEX_Z])
CODEX_AXIS = CODEX_CENTER / np.linalg.norm(CODEX_CENTER)

# ==========================================
# 2. DATA LOADING
# ==========================================
DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
BKG_LABEL = "MUON"

DEC_ID    = "38000800" if BKG_LABEL == "KL0" else "30011001"
BKG_FILE = f"{DATA_PATH}ntuple_background_{DEC_ID}.root"
SIG_FILE = f"{DATA_PATH}ntuple_signal_40114060.root"

VAR_NAMES = ['eventNumber', 'module', 'x', 'y', 'z']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

print('Reading data with custom read_root...')
bkg_df = read_root(BKG_FILE, TREE_NAME, VAR_NAMES, nrows=10_000_000)
sig_df = read_root(SIG_FILE, TREE_NAME, VAR_NAMES, nrows=10_000_000)
print('Data loaded!')

# Extract unique events for statistics
N_EVENTS = 5000
selected_bkg_ev = bkg_df['eventNumber'].unique()[:N_EVENTS]
bkg_df = bkg_df[bkg_df['eventNumber'].isin(selected_bkg_ev)]

selected_sig_ev = sig_df['eventNumber'].unique()[:N_EVENTS]
sig_df = sig_df[sig_df['eventNumber'].isin(selected_sig_ev)]

# Group by event
bkg_grouped = bkg_df.groupby('eventNumber')
sig_grouped = sig_df.groupby('eventNumber')

# ==========================================
# 3. OPTIMIZED METRIC COMPUTATION
# ==========================================
def compute_event_metrics(grouped_df):
    event_counts = []
    total_ev = len(grouped_df)
    
    for c, (event_number, event_df) in enumerate(grouped_df):
        if c % 500 == 0 and c > 0:
            print(f"  ... Processed {c}/{total_ev} events")
            
        coords = event_df[['x', 'y', 'z']].values
        mods = event_df['module'].values
        
        # FAST PAIR GENERATION
        # Map module IDs to their indices in the coordinate array
        mod_to_idx = {m: np.where(mods == m)[0] for m in np.unique(mods)}
        i_list, j_list = [], []
        
        for m in mod_to_idx:
            idx_m = mod_to_idx[m]
            for dm in [1, 2]: # adjacent modules at distance 1 or 2
                if m + dm in mod_to_idx:
                    idx_mdm = mod_to_idx[m + dm]
                    I, J = np.meshgrid(idx_m, idx_mdm, indexing='ij')
                    i_list.append(I.flatten())
                    j_list.append(J.flatten())
                    
        if not i_list:
            event_counts.append(0) # 0 tracks if no pairs found
            continue
            
        i_idx = np.concatenate(i_list)
        j_idx = np.concatenate(j_list)

        # Ensure ordering by Z (forward propagation)
        pos_i, pos_j = coords[i_idx], coords[j_idx]
        mod_i, mod_j = mods[i_idx], mods[j_idx]

        swap_mask = np.abs(pos_j[:, 2]) < np.abs(pos_i[:, 2])
        pos_i[swap_mask], pos_j[swap_mask] = pos_j[swap_mask], pos_i[swap_mask]
        mod_i[swap_mask], mod_j[swap_mask] = mod_j[swap_mask], mod_i[swap_mask]

        # Linear Extrapolation
        step_vec = (pos_j - pos_i) / (mod_j - mod_i)[:, np.newaxis]
        
        # Look ahead for confirming hits in subsequent modules
        preds = [pos_j + step_vec, pos_j + 2.0*step_vec]
        
        hit_map = {m: coords[mods == m, :2] for m in np.unique(mods)}
        is_valid_pair = np.zeros(len(i_idx), dtype=bool)

        for p_idx, pred in enumerate(preds):
            # Define target modules for extrapolation
            target_mods = (mod_j + 1) if p_idx == 0 else (mod_j + 2)
            
            for m in np.unique(target_mods):
                if m not in hit_map: continue
                m_mask = (target_mods == m)
                q_pts = pred[m_mask, :2]
                if len(q_pts) == 0: continue
                
                if HAVE_CKD:
                    tree_mod = cKDTree(hit_map[m])
                    d, _ = tree_mod.query(q_pts)
                else:
                    d = np.min(np.linalg.norm(q_pts[:, None] - hit_map[m][None, :], axis=2), axis=1)
                
                is_valid_pair[np.where(m_mask)[0][d <= EXTRAP_TOL]] = True

        if not np.any(is_valid_pair):
            event_counts.append(0) # 0 tracks if none pass validation
            continue

        # --- Extract metric for the event ---
        valid_pos_i = pos_i[is_valid_pair]
        segment_vectors = (pos_j - pos_i)[is_valid_pair]
        
        norms = np.linalg.norm(segment_vectors, axis=1)
        valid_mask = norms > 1e-6
        
        if not np.any(valid_mask): 
            event_counts.append(0)
            continue
            
        segment_vectors = segment_vectors[valid_mask]
        valid_pos_i = valid_pos_i[valid_mask]
        norms = norms[valid_mask]

        # 1. Cosine Similarity & Count > 0.9
        dots = np.dot(segment_vectors, CODEX_AXIS)
        cos_sim = dots / norms
        
        # Count how many segments have cos_sim > 0.9
        num_aligned = np.sum(cos_sim > 0.9)
        event_counts.append(num_aligned)

    return np.array(event_counts)

print('Computing alignments...')
bkg_counts = compute_event_metrics(bkg_grouped)
sig_counts = compute_event_metrics(sig_grouped)
print('Alignments computed!')

# ==========================================
# 4. PLOTTING
# ==========================================
os.makedirs("check_plots", exist_ok=True)
plt.figure(figsize=(9, 6))

# Determine max count for binning
max_count = int(max(np.max(bkg_counts), np.max(sig_counts)))


plt.hist(bkg_counts, bins=20, alpha=0.8, label=f'Background ({len(bkg_counts)} ev)', histtype='step', color='red', density=True, lw=2, range=(0, 500))
plt.hist(sig_counts, bins=20, alpha=0.8, label=f'Signal ({len(sig_counts)} ev)', histtype='step', color='blue', density=True, lw=2, range=(0, 500))

plt.xlabel('Number of tracks with cos(θ) > 0.9 per event')
plt.ylabel('density')
plt.title('Distribution of highly aligned tracks towards CODEX-b')


# Log scale is recommended as most events have 0 aligned tracks
plt.yscale('log') 

plt.legend()
plt.tight_layout()
plt.savefig("check_plots/codex_alignment_check.png")
print("\nDone! Plots saved to check_plots/")