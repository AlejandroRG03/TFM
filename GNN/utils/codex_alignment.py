"""
CODEX-b Alignment Validation Script.
This script identifies track segments (pairs of hits in adjacent modules) 
that are confirmed by a third hit and evaluates their alignment with 
the CODEX-b detector axis.
"""

import sys
import os
import multiprocessing as mp
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

N_CORES = max(1, mp.cpu_count() - 1)

# ==========================================
# 1. GEOMETRIC PARAMETERS AND FILTERS
# ==========================================
MAX_ITER = 1        # Number of sequential extrapolation steps
EXTRAP_TOL = 5    # mm (Tolerance for hit matching)
QUANTILE_CUT = 0.01
ALIGNMENT_CUT = 0.95  # cos(θ) threshold for alignment with CODEX-b axis
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

VAR_NAMES = ['eventNumber', 'module', 'x', 'y', 'z', 'beamspotX', 'beamspotY']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

print('Reading data with custom read_root...')
bkg_df = read_root(BKG_FILE, TREE_NAME, VAR_NAMES, nrows=100_000_000)
sig_df = read_root(SIG_FILE, TREE_NAME, VAR_NAMES, nrows=100_000_000)
print('Data loaded!')

# Extract unique events for statistics
N_EVENTS = 50000
selected_bkg_ev = bkg_df['eventNumber'].unique()[:N_EVENTS]
bkg_df = bkg_df[bkg_df['eventNumber'].isin(selected_bkg_ev)]

selected_sig_ev = sig_df['eventNumber'].unique()[:N_EVENTS]
sig_df = sig_df[sig_df['eventNumber'].isin(selected_sig_ev)]

# compute hit multiplicity per event and use it for filtering

bkg_df['eventMultiplicity'] = bkg_df.groupby('eventNumber')['eventNumber'].transform('count')
sig_df['eventMultiplicity'] = sig_df.groupby('eventNumber')['eventNumber'].transform('count')

print(f"Max multiplicity before cut: Background={bkg_df['eventMultiplicity'].max()}, Signal={sig_df['eventMultiplicity'].max()}")

bkg_unique = bkg_df[['eventNumber', 'eventMultiplicity']].drop_duplicates().sort_values('eventMultiplicity')
sig_unique = sig_df[['eventNumber', 'eventMultiplicity']].drop_duplicates().sort_values('eventMultiplicity')
bkg_low_events = bkg_unique.iloc[:int(len(bkg_unique) * QUANTILE_CUT)]['eventNumber']
sig_low_events = sig_unique.iloc[:int(len(sig_unique) * QUANTILE_CUT)]['eventNumber']
bkg_df = bkg_df[bkg_df['eventNumber'].isin(bkg_low_events)]
sig_df = sig_df[sig_df['eventNumber'].isin(sig_low_events)]

print(f"Selected {len(bkg_df['eventNumber'].unique())} background events and {len(sig_df['eventNumber'].unique())} signal events after quantile cut.")
print(f"Max multiplicity after cut: Background={bkg_df['eventMultiplicity'].max()}, Signal={sig_df['eventMultiplicity'].max()}")


# Shift to center on beamspot
bkg_df['x'] -= bkg_df['beamspotX']
bkg_df['y'] -= bkg_df['beamspotY']
sig_df['x'] -= sig_df['beamspotX']
sig_df['y'] -= sig_df['beamspotY']

# Group by event
bkg_grouped = bkg_df.groupby('eventNumber')
sig_grouped = sig_df.groupby('eventNumber')

# ==========================================
# 3. OPTIMIZED METRIC COMPUTATION
# ==========================================
def _process_event(event_data):
    event_number, event_df = event_data

    coords = event_df[['x', 'y', 'z']].values
    mods = event_df['module'].values

    # FAST PAIR GENERATION
    mod_to_idx = {m: np.where(mods == m)[0] for m in np.unique(mods)}
    i_list, j_list = [], []

    for m in mod_to_idx:
        idx_m = mod_to_idx[m]
        for dm in [1, 2]:
            if m + dm in mod_to_idx:
                idx_mdm = mod_to_idx[m + dm]
                i_list.append(np.repeat(idx_m, len(idx_mdm)))
                j_list.append(np.tile(idx_mdm, len(idx_m)))

    if not i_list:
        return (0, 0)

    i_idx = np.concatenate(i_list)
    j_idx = np.concatenate(j_list)

    pos_i, pos_j = coords[i_idx], coords[j_idx]
    mod_i, mod_j = mods[i_idx], mods[j_idx]

    swap_mask = np.abs(pos_j[:, 2]) < np.abs(pos_i[:, 2])
    pos_i[swap_mask], pos_j[swap_mask] = pos_j[swap_mask], pos_i[swap_mask]
    mod_i[swap_mask], mod_j[swap_mask] = mod_j[swap_mask], mod_i[swap_mask]

    # --- ITERATIVE EXTRAPOLATION ---
    hit_map = {m: coords[mods == m] for m in np.unique(mods)}
    if HAVE_CKD:
        hit_trees = {m: cKDTree(h[:, :2]) for m, h in hit_map.items()}

    mod_a, pos_a = mod_i.copy(), pos_i.copy()
    mod_b, pos_b = mod_j.copy(), pos_j.copy()
    active = np.ones(len(i_idx), dtype=bool)

    for iteration in range(MAX_ITER):
        if not np.any(active):
            break

        act = np.where(active)[0]
        dm = mod_b[act] - mod_a[act]
        step = (pos_b[act] - pos_a[act]) / dm[:, None]

        tmod = mod_b[act] + 1
        pred = pos_b[act] + step
        tol = EXTRAP_TOL * (1 + 0.5 * iteration)

        still_active = np.zeros(len(act), dtype=bool)

        for m in np.unique(tmod):
            if m not in hit_map:
                continue

            m_sel = (tmod == m)
            q_pts = pred[m_sel, :2]
            hits = hit_map[m]

            if HAVE_CKD:
                d, hit_idx = hit_trees[m].query(q_pts)
            else:
                dists = np.linalg.norm(q_pts[:, None] - hits[None, :, :2], axis=2)
                d = np.min(dists, axis=1)
                hit_idx = np.argmin(dists, axis=1)

            valid = d <= tol

            m_locals = np.where(m_sel)[0]
            valid_mask = np.zeros(len(act), dtype=bool)
            valid_mask[m_locals[valid]] = True

            gi = act[valid_mask]
            mod_a[gi] = mod_b[gi]
            pos_a[gi] = pos_b[gi]
            mod_b[gi] = m
            pos_b[gi] = hits[hit_idx[valid]]
            still_active[valid_mask] = True

        active[act] = still_active

    if not np.any(active):
        return (0, 0)

    segment_vectors = (pos_b - pos_a)[active]
    norms = np.linalg.norm(segment_vectors, axis=1)
    valid_mask = norms > 1e-6

    total_tracks = int(np.sum(valid_mask))
    if total_tracks == 0:
        return (0, 0)

    segment_vectors = segment_vectors[valid_mask]
    norms = norms[valid_mask]

    dots = np.dot(segment_vectors, CODEX_AXIS)
    cos_sim = dots / norms

    aligned_tracks = int(np.sum(cos_sim > ALIGNMENT_CUT))
    return (total_tracks, aligned_tracks)


def compute_event_metrics(grouped_df):
    events = [(ev, df) for ev, df in grouped_df]
    total_ev = len(events)
    use_mp = N_CORES > 1 and total_ev > 1

    if use_mp:
        with mp.Pool(N_CORES) as pool:
            results = []
            for i, r in enumerate(pool.imap(_process_event, events)):
                results.append(r)
                if (i + 1) % 500 == 0:
                    print(f"  ... Processed {i+1}/{total_ev} events")
    else:
        results = []
        for i, (ev, df) in enumerate(events):
            results.append(_process_event((ev, df)))
            if (i + 1) % 500 == 0:
                print(f"  ... Processed {i+1}/{total_ev} events")

    totals = np.array([r[0] for r in results])
    aligned = np.array([r[1] for r in results])
    return totals, aligned


if __name__ == '__main__':
    print('Computing alignments...')
    bkg_totals, bkg_aligned = compute_event_metrics(bkg_grouped)
    sig_totals, sig_aligned = compute_event_metrics(sig_grouped)
    print('Alignments computed!')

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    os.makedirs("check_plots", exist_ok=True)
    plt.figure(figsize=(9, 6))

    # histogram of ratios

    bkg_ratio = bkg_aligned / np.maximum(bkg_totals, 1)
    sig_ratio = sig_aligned / np.maximum(sig_totals, 1)

    common_bins = np.histogram_bin_edges(np.concatenate([bkg_ratio, sig_ratio]), bins=30)

    plt.hist(bkg_ratio, bins=common_bins, alpha=0.8, label=f'Background ({BKG_LABEL}) ({len(bkg_aligned)} ev)', histtype='step', color='red', density=True, lw=2)
    plt.hist(sig_ratio, bins=common_bins, alpha=0.8, label=f'Signal ({len(sig_aligned)} ev)', histtype='step', color='blue', density=True, lw=2)

    plt.xlabel(f'Ratio of tracks with cos(θ) > {ALIGNMENT_CUT} per event')
    plt.ylabel('density')
    plt.title(f'Distribution of highly aligned tracks towards CODEX-b \n (First {QUANTILE_CUT*100}% of event multiplicity)')

    # Log scale is recommended as most events have 0 aligned tracks
    plt.yscale('log') 

    plt.legend()
    plt.tight_layout()
    plt.savefig("check_plots/codex_alignment_check.png")
    print("\nDone! Plots saved to check_plots/codex_alignment_check.png")