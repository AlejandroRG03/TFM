import sys
import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *
from scipy.spatial import KDTree
set_tfm_style()

N_CORES = max(1, mp.cpu_count() - 1)

# --- parameters ---
MAX_ITER = 1
EXTRAP_TOL = 5
ALIGNMENT_CUT = 0.99
N_EVENTS = 10_000
MAX_XY_DIST = 30  # mm — max distance in (x,y) to pair hits between modules

CODEX_X, CODEX_Y, CODEX_Z = 23725, 0, 12650
CODEX_CENTER = np.array([CODEX_X, CODEX_Y, CODEX_Z])
CODEX_AXIS = CODEX_CENTER / np.linalg.norm(CODEX_CENTER)

# --- loading ---
DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"

SIG_IDS = ["40114060", "11114033"]
BKG_IDS = ["30011001", "38000800"]

VAR_NAMES = ['eventNumber', 'module', 'x', 'y', 'z', 'beamspotX', 'beamspotY']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

def load_and_filter(file_path, n_events):
    df = read_root(file_path, TREE_NAME, VAR_NAMES, nrows=100_000_000)
    evts = df['eventNumber'].unique()[:n_events]
    df = df[df['eventNumber'].isin(evts)]
    df['x'] -= df['beamspotX']
    df['y'] -= df['beamspotY']
    return df

print('Loading data...')
dfs_data = {}
for sid in SIG_IDS:
    dfs_data[sid] = load_and_filter(f"{DATA_PATH}ntuple_signal_{sid}.root", N_EVENTS)
for bid in BKG_IDS:
    dfs_data[bid] = load_and_filter(f"{DATA_PATH}ntuple_background_{bid}.root", N_EVENTS)
print('Data loaded!')

# --- core event processing ---
def _process_event(event_data):
    event_number, event_df = event_data
    coords = event_df[['x', 'y', 'z']].values
    mods = event_df['module'].values

    mod_to_idx = {m: np.where(mods == m)[0] for m in np.unique(mods)}
    i_list, j_list = [], []

    for m in mod_to_idx:
        idx_m = mod_to_idx[m]
        if len(idx_m) == 0:
            continue
        xy_m = coords[idx_m, :2]
        for dm in [1, 2]:
            if m + dm not in mod_to_idx:
                continue
            idx_mdm = mod_to_idx[m + dm]
            xy_mdm = coords[idx_mdm, :2]

            # Use KD-tree to find spatially close pairs
            tree = KDTree(xy_mdm)
            pairs = tree.query_ball_point(xy_m, r=MAX_XY_DIST)

            for local_i, hits_in_range in enumerate(pairs):
                if len(hits_in_range) == 0:
                    continue
                global_i = idx_m[local_i]
                for local_j in hits_in_range:
                    global_j = idx_mdm[local_j]
                    i_list.append(global_i)
                    j_list.append(global_j)

    if not i_list:
        return (0, 0)

    i_idx = np.array(i_list)
    j_idx = np.array(j_list)

    pos_i, pos_j = coords[i_idx], coords[j_idx]
    mod_i, mod_j = mods[i_idx], mods[j_idx]

    swap_mask = np.abs(pos_j[:, 2]) < np.abs(pos_i[:, 2])
    pos_i[swap_mask], pos_j[swap_mask] = pos_j[swap_mask], pos_i[swap_mask]
    mod_i[swap_mask], mod_j[swap_mask] = mod_j[swap_mask], mod_i[swap_mask]

    hit_map = {m: coords[mods == m] for m in np.unique(mods)}
    hit_trees = {m: KDTree(h[:, :2]) for m, h in hit_map.items()}

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
            tree = hit_trees[m]

            d, hit_idx = tree.query(q_pts)
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
    if not np.any(valid_mask):
        return (0, 0)

    segment_vectors = segment_vectors[valid_mask]
    norms = norms[valid_mask]
    dots = np.dot(segment_vectors, CODEX_AXIS)
    cos_sim = dots / norms
    aligned_tracks = int(np.sum(cos_sim > ALIGNMENT_CUT))
    return (int(np.sum(valid_mask)), aligned_tracks)


def compute_event_metrics(grouped_df):
    events = [(ev, df) for ev, df in grouped_df]
    total_ev = len(events)
    use_mp = N_CORES > 1 and total_ev > 1

    if use_mp:
        with mp.Pool(N_CORES) as pool:
            results = [r for r in pool.imap(_process_event, events)]
    else:
        results = [_process_event(e) for e in events]

    totals = np.array([r[0] for r in results])
    aligned = np.array([r[1] for r in results])
    return totals, aligned


# --- compute for all datasets ---
sig_names = {"40114060": "Dark Photon", "11114033": "Dark Higgs"}
bkg_names = {"30011001": "MUON", "38000800": "KL0"}

datasets = {}
for sid in SIG_IDS:
    print(f'Processing signal {sig_names[sid]} ({sid})...')
    grp = dfs_data[sid].groupby('eventNumber')
    totals, aligned = compute_event_metrics(grp)
    datasets[sid] = aligned
    print(f'  {len(aligned)} events processed')

for bid in BKG_IDS:
    print(f'Processing background {bkg_names[bid]} ({bid})...')
    grp = dfs_data[bid].groupby('eventNumber')
    totals, aligned = compute_event_metrics(grp)
    datasets[bid] = aligned
    print(f'  {len(aligned)} events processed')

# --- plot 1x4 ---
combinaciones = [
    ("40114060", "30011001", "Dark Photon", "MUON"),
    ("40114060", "38000800", "Dark Photon", "KL0"),
    ("11114033", "30011001", "Dark Higgs", "MUON"),
    ("11114033", "38000800", "Dark Higgs", "KL0"),
]

bins = np.linspace(0, 50, 51)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for idx, (sid, bid, sig_label, bkg_label) in enumerate(combinaciones):
    ax = axes[idx]
    sig_counts = datasets[sid]
    bkg_counts = datasets[bid]

    ax.hist(sig_counts, bins=bins, histtype='step', color='blue',
            label=sig_label, density=True)
    ax.hist(bkg_counts, bins=bins, histtype='step', color='red',
            label=bkg_label, density=True)

    ax.set_xlabel('Aligned tracks per event')
    ax.set_ylabel('PDF')
    ax.set_yscale('log')
    ax.legend(fontsize=10)

fig.subplots_adjust(hspace=0.05, wspace=0.3, left=0.08, right=0.98, top=0.92, bottom=0.3)
fig.savefig('plots_tfm/track_alignment.pdf', bbox_inches='tight', pad_inches=0.5)
print('Saved plots_tfm/track_alignment.pdf')
