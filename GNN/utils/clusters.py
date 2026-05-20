import os
import sys
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home3/alejandro.rodriguez")
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *
set_tfm_style()

ALIGNMENT_CUT = 0.95
N_CORES = max(1, mp.cpu_count() - 1)

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ==========================================
# 1. DATA LOADING
# ==========================================
DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
BKG_LABEL = "MUON"

DEC_ID    = "38000800" if BKG_LABEL == "KL0" else "30011001"
BKG_FILE = f"{DATA_PATH}ntuple_background_{DEC_ID}.root"
SIG_FILE = f"{DATA_PATH}ntuple_signal_40114060.root"

VAR_NAMES = ['eventNumber', 'module', 'x', 'y', 'z', 'beamspotX', 'beamspotY']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

print('Reading data with custom read_root...')
bkg_df = read_root(BKG_FILE, TREE_NAME, VAR_NAMES, nrows=30_000_000)
sig_df = read_root(SIG_FILE, TREE_NAME, VAR_NAMES, nrows=30_000_000)
print('Data loaded!')

# Shift to center on beamspot
bkg_df['x'] -= bkg_df['beamspotX']
bkg_df['y'] -= bkg_df['beamspotY']
sig_df['x'] -= sig_df['beamspotX']
sig_df['y'] -= sig_df['beamspotY']

# group by event, drop last (may be incomplete)
bkg_events = [g for _, g in bkg_df.groupby('eventNumber')][:-1]
sig_events = [g for _, g in sig_df.groupby('eventNumber')][:-1]
print(f"Loaded {len(bkg_events)} background events, {len(sig_events)} signal events")

# ==========================================
# 2. CODEX GEOMETRIC DEFINITION
# ==========================================
CODEX_X, CODEX_Y, CODEX_Z = 23725, 0, 12650
CODEX_L = 10_000
CODEX_Z_FRONT = CODEX_Z - CODEX_L / 2

CODEX_CENTER = np.array([CODEX_X, CODEX_Y, CODEX_Z])
CODEX_AXIS = CODEX_CENTER / np.linalg.norm(CODEX_CENTER)

# ==========================================
# 3. GEOMETRIC CLUSTERING
# ==========================================

def cluster_hits(coords, eps_angle=0.03, min_samples=2):
    n = len(coords)
    if n == 0:
        return np.array([], dtype=int), 0

    norms = np.linalg.norm(coords, axis=1)
    dirs = coords / norms[:, None]

    chunk_dist = 2 * np.sin(eps_angle / 2)
    tree = cKDTree(dirs)
    adj = tree.query_ball_tree(tree, r=chunk_dist)

    rows, cols = [], []
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            if i < j:
                rows.append(i)
                cols.append(j)

    if not rows:
        return np.full(n, -1, dtype=int), 0

    data = np.ones(len(rows), dtype=bool)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph + graph.T

    n_comp, labels = connected_components(csgraph=graph, directed=False)

    sizes = np.bincount(labels)
    for cid in range(n_comp):
        if sizes[cid] < min_samples:
            labels[labels == cid] = -1

    n_clusters = np.sum(sizes >= min_samples)
    return labels, n_clusters


def cluster_direction(cluster_coords):
    if len(cluster_coords) < 2:
        return None
    centroid = cluster_coords.mean(axis=0)
    centered = cluster_coords - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[0]


def _process_event(event_df):
    coords = event_df[['x', 'y', 'z']].values

    labels, n_clusters = cluster_hits(coords)

    if n_clusters == 0:
        return 0, 0, np.array([]), np.array([])

    aligned = 0
    cluster_sizes = []
    cluster_cos = []

    for cid in range(labels.max() + 1):
        mask = labels == cid
        size = int(np.sum(mask))
        if size < 2:
            continue
        cluster_sizes.append(size)

        direction = cluster_direction(coords[mask])
        if direction is None:
            continue

        cos_sim = abs(np.dot(direction, CODEX_AXIS))
        cluster_cos.append(cos_sim)
        if cos_sim > ALIGNMENT_CUT:
            aligned += 1

    return n_clusters, aligned, np.array(cluster_sizes), np.array(cluster_cos)


def compute_event_metrics(events):
    total = len(events)
    use_mp = N_CORES > 1 and total > 1

    if use_mp:
        with mp.Pool(N_CORES) as pool:
            results = []
            for i, r in enumerate(pool.imap(_process_event, events)):
                results.append(r)
                if (i + 1) % 200 == 0:
                    print(f"  ... Processed {i+1}/{total} events")
    else:
        results = []
        for i, ev in enumerate(events):
            results.append(_process_event(ev))
            if (i + 1) % 200 == 0:
                print(f"  ... Processed {i+1}/{total} events")

    n_clusters = np.array([r[0] for r in results])
    aligned = np.array([r[1] for r in results])
    all_sizes = np.concatenate([r[2] for r in results if len(r[2]) > 0])
    all_cos = np.concatenate([r[3] for r in results if len(r[3]) > 0])
    return n_clusters, aligned, all_sizes, all_cos


print('Computing geometric clustering...')
bkg_ncl, bkg_aligned, bkg_sizes, bkg_cos = compute_event_metrics(bkg_events)
sig_ncl, sig_aligned, sig_sizes, sig_cos = compute_event_metrics(sig_events)
print('Computation done!')

# ==========================================
# 4. PLOTTING
# ==========================================
os.makedirs("check_plots", exist_ok=True)

# Ratio of aligned clusters per event
bkg_ratio = bkg_aligned / np.maximum(bkg_ncl, 1)
sig_ratio = sig_aligned / np.maximum(sig_ncl, 1)

plt.figure(figsize=(9, 6))
common_bins = np.histogram_bin_edges(np.concatenate([bkg_ratio, sig_ratio]), bins=20)
plt.hist(bkg_ratio, bins=common_bins, alpha=0.8, label=f'Background ({BKG_LABEL}) ({len(bkg_ncl)} ev)', histtype='step', color='red', density=True, lw=2)
plt.hist(sig_ratio, bins=common_bins, alpha=0.8, label=f'Signal ({len(sig_ncl)} ev)', histtype='step', color='blue', density=True, lw=2)
plt.xlabel(f'Ratio of clusters with cos(θ) > {ALIGNMENT_CUT} per event')
plt.ylabel('Density')
plt.title('Distribution of highly aligned clusters towards CODEX-b')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("check_plots/cluster_alignment_ratio_hist.png")
print("Saved cluster_alignment_ratio_hist.png")

# Cluster cos similarity distribution
plt.figure(figsize=(9, 6))
common_bins = np.histogram_bin_edges(np.concatenate([bkg_cos, sig_cos]), bins=30)
plt.hist(bkg_cos, bins=common_bins, alpha=0.8, label=f'Background ({BKG_LABEL})', histtype='step', color='red', density=True, lw=2)
plt.hist(sig_cos, bins=common_bins, alpha=0.8, label='Signal', histtype='step', color='blue', density=True, lw=2)
plt.xlabel('Cosine similarity of cluster direction with CODEX-b axis')
plt.ylabel('Density')
plt.title('Distribution of cluster alignment towards CODEX-b')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("check_plots/cluster_cosine_similarity_hist.png")
print("Saved cluster_cosine_similarity_hist.png")

# Cluster size distribution
plt.figure(figsize=(9, 6))
common_bins = np.histogram_bin_edges(np.concatenate([bkg_sizes, sig_sizes]), bins=20)
plt.hist(bkg_sizes, bins=common_bins, alpha=0.8, label=f'Background ({BKG_LABEL})', histtype='step', color='red', density=True, lw=2)
plt.hist(sig_sizes, bins=common_bins, alpha=0.8, label='Signal', histtype='step', color='blue', density=True, lw=2)
plt.xlabel('Cluster size (number of hits)')
plt.ylabel('Density')
plt.title('Distribution of cluster sizes')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("check_plots/cluster_size_hist.png")
print("Saved cluster_size_hist.png")

print("\nDone!")
