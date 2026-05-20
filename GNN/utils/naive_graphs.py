"""
Carga grafos ya construidos y explora la separabilidad del problema.
Genera todos los plots relevantes en check_plots/naive_graph_checks/

Uso:
  python naive_graphs.py [--chunk 1] [--max_events 200]
"""

import os, sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import set_tfm_style

import torch
from torch_geometric.data import Data

set_tfm_style()

SIGNAL_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060"
MUON_DIR   = "/lustre/LHCb/alejandro.rodriguez/torch_data/background/30011001"
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_plots", "naive_graph_checks")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_chunk(dirpath, chunk_id, label=None):
    path = os.path.join(dirpath, f"graphs_{chunk_id}.pt")
    print(f"[{label or '?'}] Loading {path}")
    raw = torch.load(path, weights_only=False, map_location="cpu", mmap=True)

    if any(k.startswith('data.') for k in raw):
        data_keys = [k[5:] for k in raw if k.startswith('data.')]
        num_graphs = raw['slices.y'].size(0) - 1
        graphs = []
        for i in range(num_graphs):
            g = Data()
            for key in data_keys:
                s = raw[f'slices.{key}']
                s0, s1 = int(s[i]), int(s[i + 1])
                tensor = raw[f'data.{key}']
                if key == 'edge_index':
                    g[key] = tensor[:, s0:s1].clone().long()
                elif key == 'x_cat':
                    g[key] = tensor[s0:s1].clone().long()
                else:
                    g[key] = tensor[s0:s1].clone()
            g.num_nodes = g.x_cont.size(0)
            graphs.append(g)
        return graphs

    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def compute_degree(edge_index, num_nodes):
    return torch.bincount(edge_index[0], minlength=num_nodes).float().numpy()


# ── feature metadata ─────────────────────────────────────────────────────────

CONT_COLS = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle', 'module_side', 'module_occupancy_norm', 'degree_norm']
CONT_LABELS = ['x', 'y', 'z', r'$r_T$', r'$\phi$', r'$\eta$', 'n_pix', r'$\theta_{CODEX}$', 'module_side', 'occupancy_norm', 'degree_norm']
CONT_SHORT  = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle', 'module_side', 'occupancy_norm', 'degree_norm']

EDGE_COLS = ['dx_cm', 'dy_cm', 'dz_cm', 'dist_cm', 'd_rT', 'd_phi', 'd_zn', 'ux', 'uy', 'uz']
EDGE_LABELS = [r'$\Delta x$ (cm)', r'$\Delta y$ (cm)', r'$\Delta z$ (cm)', r'$d_{3D}$ (cm)',
               r'$\Delta r_T$', r'$\Delta\phi$', r'$\Delta z_n$', r'$u_x$', r'$u_y$', r'$u_z$']

N_FEAT = len(CONT_COLS)
N_EFEAT = len(EDGE_COLS)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=1, help="Chunk number (default: 1)")
    parser.add_argument("--signal_dir", default=SIGNAL_DIR)
    parser.add_argument("--muon_dir",   default=MUON_DIR)
    parser.add_argument("--max_events", type=int, default=200, help="Max events per class (default: 200)")
    args = parser.parse_args()

    sig_list = load_chunk(args.signal_dir, args.chunk, label="SIGNAL")
    muo_list = load_chunk(args.muon_dir,   args.chunk, label="MUON")

    if args.max_events is not None and args.max_events < len(sig_list):
        sig_list = sig_list[:args.max_events]
        muo_list = muo_list[:args.max_events]

    print(f"\n  Signal: {len(sig_list)} events")
    print(f"  Muon:   {len(muo_list)} events\n")

    # ────────────────────────────────────────────────────────────────────────
    # Build per-event stats
    # ────────────────────────────────────────────────────────────────────────
    def extract_stats(data_list):
        stats = {}
        n_nodes = np.array([d.num_nodes for d in data_list])
        n_edges = np.array([d.edge_index.shape[1] for d in data_list])
        stats['n_nodes'] = n_nodes
        stats['n_edges'] = n_edges
        stats['density'] = n_edges / (n_nodes * (n_nodes - 1)).clip(min=1)

        edge_lens = []
        for d in data_list:
            pos = d.pos
            src, dst = d.edge_index
            dists = (pos[src] - pos[dst]).norm(dim=1).numpy()
            edge_lens.append(dists)
        stats['edge_len_mean'] = np.array([e.mean() for e in edge_lens])
        stats['edge_len_std']  = np.array([e.std()  for e in edge_lens])
        stats['edge_len_pool'] = np.concatenate(edge_lens) if edge_lens else np.array([])

        ga = np.stack([d.global_attr.squeeze(0).numpy() for d in data_list])
        stats['nVtx'] = ga[:, 0]
        stats['nClu'] = ga[:, 1]
        stats['nTrk'] = ga[:, 2]

        x_mean = np.stack([d.x_cont.mean(dim=0).numpy() for d in data_list], axis=0)
        x_std  = np.stack([d.x_cont.std(dim=0).numpy()  for d in data_list], axis=0)
        stats['x_mean'] = x_mean
        stats['x_std']  = x_std

        # per-event edge attr means
        e_mean = np.stack([d.edge_attr.mean(dim=0).numpy() for d in data_list], axis=0)
        e_std  = np.stack([d.edge_attr.std(dim=0).numpy()  for d in data_list], axis=0)
        stats['e_mean'] = e_mean
        stats['e_std']  = e_std

        # per-node degree (raw, not normalised)
        degrees = [compute_degree(d.edge_index, d.num_nodes) for d in data_list]
        stats['deg_pool'] = np.concatenate(degrees)
        stats['deg_mean'] = np.array([d.mean() for d in degrees])

        # module distribution
        mods = [d.x_cat.numpy() for d in data_list]
        stats['mod_pool'] = np.concatenate(mods)

        # intra vs inter edge proportion
        intra_fracs = []
        for d in data_list:
            src_mod = d.x_cat[d.edge_index[0]]
            dst_mod = d.x_cat[d.edge_index[1]]
            intra_fracs.append((src_mod == dst_mod).float().mean().item())
        stats['intra_frac'] = np.array(intra_fracs)

        return stats

    sig = extract_stats(sig_list)
    muo = extract_stats(muo_list)

    # ────────────────────────────────────────────────────────────────────────
    # Plotting helpers
    # ────────────────────────────────────────────────────────────────────────
    def hist2(ax, a, b, bins=80, xlabel="", log=False):
        ax.hist(a, bins=bins, alpha=0.5, density=True, label='Signal', histtype='stepfilled')
        ax.hist(b, bins=bins, alpha=0.5, density=True, label='Muon', histtype='stepfilled')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        if log: ax.set_yscale('log')

    def boxplot_feature(ax, sig_vals, muo_vals, xticklabel=""):
        ax.boxplot([sig_vals, muo_vals], tick_labels=['Signal', 'Muon'], widths=0.5)
        if xticklabel: ax.set_xlabel(xticklabel)

    def save(fig, name):
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {name}")

    # ── 0. Summary table ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Quantity':<40} {'Signal':>12} {'Muon':>12}")
    print(f"{'─'*70}")
    for name, s, m in [
        ("Num events", len(sig_list), len(muo_list)),
        ("Nodes/event", f"{sig['n_nodes'].mean():.1f} ± {sig['n_nodes'].std():.1f}",
                        f"{muo['n_nodes'].mean():.1f} ± {muo['n_nodes'].std():.1f}"),
        ("Edges/event", f"{sig['n_edges'].mean():.0f} ± {sig['n_edges'].std():.0f}",
                        f"{muo['n_edges'].mean():.0f} ± {muo['n_edges'].std():.0f}"),
        ("Edge density", f"{sig['density'].mean():.6f}", f"{muo['density'].mean():.6f}"),
        ("Mean edge len (mm)", f"{sig['edge_len_mean'].mean():.2f}", f"{muo['edge_len_mean'].mean():.2f}"),
        ("Intra-module edges frac", f"{sig['intra_frac'].mean():.3f}", f"{muo['intra_frac'].mean():.3f}"),
        ("nVtx/event", f"{sig['nVtx'].mean():.2f}", f"{muo['nVtx'].mean():.2f}"),
        ("nClu/event", f"{sig['nClu'].mean():.0f}", f"{muo['nClu'].mean():.0f}"),
        ("nTrk/event", f"{sig['nTrk'].mean():.2f}", f"{muo['nTrk'].mean():.2f}"),
    ]:
        print(f"  {name:<38} {s:>12} {m:>12}")
    print(f"{'─'*70}")

    # ── 1. Graph-level stats ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    ax = axes.ravel()
    hist2(ax[0], sig['n_nodes'], muo['n_nodes'], xlabel='Nodes per graph')
    hist2(ax[1], sig['n_edges'], muo['n_edges'], xlabel='Edges per graph')
    hist2(ax[2], sig['density'], muo['density'], xlabel='Edge density')
    hist2(ax[3], sig['edge_len_mean'], muo['edge_len_mean'], xlabel='Mean edge length (mm)')
    hist2(ax[4], sig['intra_frac'], muo['intra_frac'], xlabel='Intra-module edge fraction')
    hist2(ax[5], sig['nVtx'], muo['nVtx'], xlabel='nVtx (norm)')
    hist2(ax[6], sig['nClu'], muo['nClu'], xlabel='nClu (norm)')
    hist2(ax[7], sig['nTrk'], muo['nTrk'], xlabel='nTrk (norm)')
    fig.suptitle(f'Graph-level stats — chunk {args.chunk}, {args.max_events} evts/class', fontsize=14)
    fig.tight_layout()
    save(fig, '01_graph_level_stats.pdf')

    # ── 2. n_nodes vs n_edges scatter ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sig['n_nodes'], sig['n_edges'], alpha=0.6, label='Signal', s=20)
    ax.scatter(muo['n_nodes'], muo['n_edges'], alpha=0.6, label='Muon', s=20)
    ax.set_xlabel('Nodes per graph')
    ax.set_ylabel('Edges per graph')
    ax.legend()
    fig.tight_layout()
    save(fig, '02_nodes_vs_edges_scatter.pdf')

    # ── 3. Per-event node feature means — boxplots ───────────────────────
    n_rows = int(np.ceil(N_FEAT / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
    axes = axes.ravel()
    for i in range(N_FEAT):
        boxplot_feature(axes[i], sig['x_mean'][:, i], muo['x_mean'][:, i])
        axes[i].set_title(CONT_LABELS[i])
    for i in range(N_FEAT, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Per-event mean node features — boxplots', fontsize=14)
    fig.tight_layout()
    save(fig, '03_node_mean_boxplots.pdf')

    # ── 4. Per-event node feature STDs — boxplots ────────────────────────
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
    axes = axes.ravel()
    for i in range(N_FEAT):
        boxplot_feature(axes[i], sig['x_std'][:, i], muo['x_std'][:, i])
        axes[i].set_title(CONT_LABELS[i])
    for i in range(N_FEAT, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Per-event std node features — boxplots', fontsize=14)
    fig.tight_layout()
    save(fig, '04_node_std_boxplots.pdf')

    # ── 5. Pooled node features — histograms ─────────────────────────────
    sig_nodes = torch.cat([d.x_cont for d in sig_list], dim=0).numpy()
    muo_nodes = torch.cat([d.x_cont for d in muo_list], dim=0).numpy()

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
    axes = axes.ravel()
    for i in range(N_FEAT):
        hist2(axes[i], sig_nodes[:, i], muo_nodes[:, i], bins=100, xlabel=CONT_LABELS[i],
              log=(i in (6, 7, 9, 10)))
    for i in range(N_FEAT, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f'Node feature distributions (pooled, {len(sig_nodes)+len(muo_nodes):.0f} nodes)', fontsize=14)
    fig.tight_layout()
    save(fig, '05_node_features_pooled.pdf')

    # ── 6. Per-event edge attr means — boxplots ──────────────────────────
    n_erows = int(np.ceil(N_EFEAT / 4))
    fig, axes = plt.subplots(n_erows, 4, figsize=(16, 3.5 * n_erows))
    axes = axes.ravel()
    for i in range(N_EFEAT):
        boxplot_feature(axes[i], sig['e_mean'][:, i], muo['e_mean'][:, i])
        axes[i].set_title(EDGE_LABELS[i])
    for i in range(N_EFEAT, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Per-event mean edge attributes — boxplots', fontsize=14)
    fig.tight_layout()
    save(fig, '06_edge_mean_boxplots.pdf')

    # ── 7. Pooled edge features — histograms (subsampled) ────────────────
    np.random.seed(42)
    sig_edges = np.concatenate([d.edge_attr.numpy() for d in sig_list], axis=0)
    muo_edges = np.concatenate([d.edge_attr.numpy() for d in muo_list], axis=0)
    MAX_EDGES = 200_000
    if sig_edges.shape[0] > MAX_EDGES:
        idx = np.random.choice(sig_edges.shape[0], MAX_EDGES, replace=False)
        sig_edges = sig_edges[idx]
    if muo_edges.shape[0] > MAX_EDGES:
        idx = np.random.choice(muo_edges.shape[0], MAX_EDGES, replace=False)
        muo_edges = muo_edges[idx]

    fig, axes = plt.subplots(n_erows, 4, figsize=(16, 3.5 * n_erows))
    axes = axes.ravel()
    for i in range(N_EFEAT):
        hist2(axes[i], sig_edges[:, i], muo_edges[:, i], bins=100, xlabel=EDGE_LABELS[i],
              log=(i in (4, 5, 6)))
    for i in range(N_EFEAT, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f'Edge feature distributions (pooled, subsampled to {MAX_EDGES:,}/class)', fontsize=14)
    fig.tight_layout()
    save(fig, '07_edge_features_pooled.pdf')

    # ── 8. Edge length distribution (pooled) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    hist2(ax, sig['edge_len_pool'], muo['edge_len_pool'], bins=150, xlabel='Edge length (mm)')
    ax.set_title('Pooled edge lengths')
    fig.tight_layout()
    save(fig, '08_edge_lengths_pooled.pdf')

    # ── 9. Edge length per-event means ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    hist2(ax, sig['edge_len_mean'], muo['edge_len_mean'], bins=60, xlabel='Mean edge length per event (mm)')
    ax.set_title('Per-event mean edge length')
    fig.tight_layout()
    save(fig, '09_edge_len_per_event.pdf')

    # ── 10. Degree distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    hist2(ax, sig['deg_pool'], muo['deg_pool'], bins=100, xlabel='Node degree (raw)')
    ax.set_title('Node degree distribution (pooled)')
    fig.tight_layout()
    save(fig, '10_degree_distribution.pdf')

    # ── 11. Module distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    sig_mod_vals, sig_mod_cnts = np.unique(sig['mod_pool'], return_counts=True)
    muo_mod_vals, muo_mod_cnts = np.unique(muo['mod_pool'], return_counts=True)
    all_mods = np.union1d(sig_mod_vals, muo_mod_vals)
    sig_cnt = np.array([sig_mod_cnts[sig_mod_vals == m].sum() if m in sig_mod_vals else 0 for m in all_mods])
    muo_cnt = np.array([muo_mod_cnts[muo_mod_vals == m].sum() if m in muo_mod_vals else 0 for m in all_mods])
    x = np.arange(len(all_mods))
    w = 0.35
    ax.bar(x - w/2, sig_cnt / sig_cnt.sum(), w, alpha=0.7, label='Signal')
    ax.bar(x + w/2, muo_cnt / muo_cnt.sum(), w, alpha=0.7, label='Muon')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(m)) for m in all_mods], rotation=45, fontsize=8)
    ax.set_xlabel('Module ID')
    ax.set_ylabel('Fraction')
    ax.legend()
    ax.set_title('Hit distribution across modules')
    fig.tight_layout()
    save(fig, '11_module_distribution.pdf')

    # ── 12. Correlation matrix — signal ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_sig = np.corrcoef(sig_nodes.T)
    mask = np.triu(np.ones_like(corr_sig, dtype=bool), k=1)
    sns.heatmap(corr_sig, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=CONT_SHORT, yticklabels=CONT_SHORT, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Node feature correlations — Signal')
    fig.tight_layout()
    save(fig, '12_corr_matrix_signal.pdf')

    # ── 13. Correlation matrix — muon ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_muo = np.corrcoef(muo_nodes.T)
    sns.heatmap(corr_muo, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=CONT_SHORT, yticklabels=CONT_SHORT, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Node feature correlations — Muon')
    fig.tight_layout()
    save(fig, '13_corr_matrix_muon.pdf')

    # ── 14. Correlation difference (signal - muon) ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_diff = corr_sig - corr_muo
    sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=CONT_SHORT, yticklabels=CONT_SHORT, vmin=-0.5, vmax=0.5, ax=ax)
    ax.set_title('Correlation difference (Signal - Muon)')
    fig.tight_layout()
    save(fig, '14_corr_difference.pdf')

    # ── 15. KS test — node features (pooled) ─────────────────────────────
    ks_stats = []
    for i in range(N_FEAT):
        stat, pval = ks_2samp(sig_nodes[:, i], muo_nodes[:, i])
        ks_stats.append((CONT_SHORT[i], stat, pval))
    ks_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*55}")
    print(f"  KS test (pooled node features — signal vs muon)")
    print(f"{'─'*55}")
    print(f"  {'Feature':<22} {'D-stat':>10} {'p-value':>12}")
    print(f"{'─'*55}")
    for name, stat, pval in ks_stats:
        p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"  {name:<22} {stat:>10.4f} {p_str:>12}")
    print(f"{'─'*55}")

    # ── 16. KS test — edge features (pooled) ─────────────────────────────
    ks_edges = []
    for i in range(N_EFEAT):
        stat, pval = ks_2samp(sig_edges[:, i], muo_edges[:, i])
        ks_edges.append((EDGE_COLS[i], stat, pval))
    ks_edges.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*55}")
    print(f"  KS test (pooled edge features — signal vs muon)")
    print(f"{'─'*55}")
    print(f"  {'Feature':<22} {'D-stat':>10} {'p-value':>12}")
    print(f"{'─'*55}")
    for name, stat, pval in ks_edges:
        p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"  {name:<22} {stat:>10.4f} {p_str:>12}")
    print(f"{'─'*55}")

    # ── 17. KS test — per-event means ────────────────────────────────────
    ks_means = []
    for i in range(N_FEAT):
        stat, pval = ks_2samp(sig['x_mean'][:, i], muo['x_mean'][:, i])
        ks_means.append((CONT_SHORT[i], stat, pval))
    ks_means.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*55}")
    print(f"  KS test (per-event mean node features)")
    print(f"{'─'*55}")
    print(f"  {'Feature':<22} {'D-stat':>10} {'p-value':>12}")
    print(f"{'─'*55}")
    for name, stat, pval in ks_means:
        p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"  {name:<22} {stat:>10.4f} {p_str:>12}")
    print(f"{'─'*55}")

    # ── 18. KS test bar chart (node features, pooled) ────────────────────
    names = [x[0] for x in ks_stats]
    vals  = [x[1] for x in ks_stats]
    colors = ['crimson' if v > 0.3 else 'orange' if v > 0.15 else 'steelblue' for v in vals]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='k')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('KS D-statistic')
    ax.set_title('Separability of pooled node features (KS test: Signal vs Muon)')
    ax.axvline(0.3, ls='--', color='gray', alpha=0.5, label='D=0.3 (strong)')
    ax.axvline(0.15, ls=':', color='gray', alpha=0.5, label='D=0.15 (moderate)')
    ax.legend()
    fig.tight_layout()
    save(fig, '15_ks_node_features_pooled.pdf')

    # ── 19. KS test bar chart (edge features, pooled) ────────────────────
    enames = [x[0] for x in ks_edges]
    evals  = [x[1] for x in ks_edges]
    ecolors = ['crimson' if v > 0.3 else 'orange' if v > 0.15 else 'steelblue' for v in evals]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(len(enames)), evals, color=ecolors, edgecolor='k')
    ax.set_yticks(range(len(enames)))
    ax.set_yticklabels(enames)
    ax.set_xlabel('KS D-statistic')
    ax.set_title('Separability of pooled edge features (KS test: Signal vs Muon)')
    ax.axvline(0.3, ls='--', color='gray', alpha=0.5, label='D=0.3 (strong)')
    ax.axvline(0.15, ls=':', color='gray', alpha=0.5, label='D=0.15 (moderate)')
    ax.legend()
    fig.tight_layout()
    save(fig, '16_ks_edge_features_pooled.pdf')

    # ── 20. KS test bar chart (per-event means) ──────────────────────────
    mnames = [x[0] for x in ks_means]
    mvals  = [x[1] for x in ks_means]
    mcolors = ['crimson' if v > 0.5 else 'orange' if v > 0.3 else 'steelblue' for v in mvals]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(len(mnames)), mvals, color=mcolors, edgecolor='k')
    ax.set_yticks(range(len(mnames)))
    ax.set_yticklabels(mnames)
    ax.set_xlabel('KS D-statistic')
    ax.set_title('Separability of per-event mean node features (KS test: Signal vs Muon)')
    ax.axvline(0.5, ls='--', color='gray', alpha=0.5, label='D=0.5 (strong)')
    ax.axvline(0.3, ls=':', color='gray', alpha=0.5, label='D=0.3 (moderate)')
    ax.legend()
    fig.tight_layout()
    save(fig, '17_ks_node_means.pdf')

    # ── 21. UMAP on per-event mean features ──────────────────────────────
    try:
        import umap
        combined = np.vstack([sig['x_mean'], muo['x_mean']])
        labels = np.array([0]*len(sig['x_mean']) + [1]*len(muo['x_mean']))
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        emb = reducer.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(emb[labels==0, 0], emb[labels==0, 1], alpha=0.7, label='Signal', s=15)
        ax.scatter(emb[labels==1, 0], emb[labels==1, 1], alpha=0.7, label='Muon', s=15)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.legend()
        ax.set_title('UMAP on per-event mean node features')
        fig.tight_layout()
        save(fig, '18_umap_node_means.pdf')
        del reducer, combined
    except ImportError:
        print("  [SKIP] UMAP not installed. Install with: pip install umap-learn")

    # ── 22. PCA on per-event mean features ───────────────────────────────
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    combined = np.vstack([sig['x_mean'], muo['x_mean']])
    labels = np.array([0]*len(sig['x_mean']) + [1]*len(muo['x_mean']))
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(combined_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(emb_pca[labels==0, 0], emb_pca[labels==0, 1], alpha=0.7, label='Signal', s=20)
    ax.scatter(emb_pca[labels==1, 0], emb_pca[labels==1, 1], alpha=0.7, label='Muon', s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    ax.set_title('PCA on per-event mean node features')

    ax = axes[1]
    components = pca.components_.T
    y_pos = np.arange(len(CONT_SHORT))
    ax.barh(y_pos, components[:, 0], color='steelblue', alpha=0.7, label='PC1')
    ax.barh(y_pos, components[:, 1], color='crimson', alpha=0.7, label='PC2', left=components[:, 0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CONT_SHORT)
    ax.set_xlabel('Loading')
    ax.axvline(0, color='k', lw=0.5)
    ax.legend(fontsize=10)
    ax.set_title('PCA loadings')
    fig.tight_layout()
    save(fig, '19_pca_node_means.pdf')

    # ── 23. Most discriminative 2D scatter pairs ─────────────────────────
    top4 = [x[0] for x in ks_means[:4]]
    top4_idx = [CONT_SHORT.index(f) for f in top4]
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for pi, (i, j) in enumerate(pairs):
        ax = axes[pi]
        ax.scatter(sig['x_mean'][:, top4_idx[i]], sig['x_mean'][:, top4_idx[j]],
                   alpha=0.6, label='Signal', s=15, c='C0')
        ax.scatter(muo['x_mean'][:, top4_idx[i]], muo['x_mean'][:, top4_idx[j]],
                   alpha=0.6, label='Muon', s=15, c='C3')
        ax.set_xlabel(f'Mean {top4[i]}')
        ax.set_ylabel(f'Mean {top4[j]}')
        ax.legend(fontsize=10)
    fig.suptitle('Top-4 discriminative features (per-event means) — pairwise scatter', fontsize=14)
    fig.tight_layout()
    save(fig, '20_top_features_scatter.pdf')

    # ── 24. Example graph — adjacency sparsity pattern ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax_idx, (d_list, label, color) in enumerate([(sig_list, 'Signal', 'C0'), (muo_list, 'Muon', 'C3')]):
        ax = axes[ax_idx]
        d = d_list[0]
        n = min(d.num_nodes, 500)
        adj = torch.zeros((n, n), dtype=torch.bool)
        ei = d.edge_index
        mask = (ei[0] < n) & (ei[1] < n)
        adj[ei[0, mask], ei[1, mask]] = True
        ax.spy(adj, markersize=0.5, color=color)
        ax.set_title(f'{label}: first {n} nodes, {adj.sum().item()} edges')
    fig.suptitle('Adjacency sparsity pattern (first 500 nodes)', fontsize=14)
    fig.tight_layout()
    save(fig, '21_adjacency_sparsity.pdf')

    # ── 25. Scatter: event size vs feature means ─────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    interesting = ['mean degree_norm', 'mean phi', 'mean r_T', 'mean eta', 'mean codex_angle', 'mean n_pix']
    interesting_idx = [CONT_SHORT.index(s.split()[-1]) for s in interesting]
    for i, (idx, title) in enumerate(zip(interesting_idx, interesting)):
        ax = axes[i]
        ax.scatter(sig['n_nodes'], sig['x_mean'][:, idx], alpha=0.6, label='Signal', s=15)
        ax.scatter(muo['n_nodes'], muo['x_mean'][:, idx], alpha=0.6, label='Muon', s=15)
        ax.set_xlabel('Nodes per event')
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
    fig.suptitle('Event size vs feature means', fontsize=14)
    fig.tight_layout()
    save(fig, '22_event_size_vs_features.pdf')

    # ── 26. global_attr vs n_nodes ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, (name, s_vals, m_vals) in enumerate([
        ('nVtx (norm)', sig['nVtx'], muo['nVtx']),
        ('nClu (norm)', sig['nClu'], muo['nClu']),
        ('nTrk (norm)', sig['nTrk'], muo['nTrk']),
    ]):
        ax = axes[ax_idx]
        ax.scatter(s_vals, sig['n_nodes'], alpha=0.6, label='Signal', s=15)
        ax.scatter(m_vals, muo['n_nodes'], alpha=0.6, label='Muon', s=15)
        ax.set_xlabel(name)
        ax.set_ylabel('Nodes per event')
        ax.legend(fontsize=9)
    fig.suptitle('Global attributes vs event size', fontsize=14)
    fig.tight_layout()
    save(fig, '23_global_attr_vs_nodes.pdf')

    # ── 27. Per-event edge attr means (top KS edge features) ─────────────
    top4_edge = [x[0] for x in ks_edges[:4]]
    top4_eidx = [EDGE_COLS.index(f) for f in top4_edge]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for pi, idx in enumerate(top4_eidx):
        ax = axes[pi]
        boxplot_feature(ax, sig['e_mean'][:, idx], muo['e_mean'][:, idx])
        ax.set_title(f'Mean {EDGE_LABELS[idx]}')
    fig.suptitle('Top-4 most separating edge features (per-event mean)', fontsize=14)
    fig.tight_layout()
    save(fig, '24_top_edge_means_boxplots.pdf')

    # ── 28. Pooled degree distribution (log-log) ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sig_deg_vals, sig_deg_cnts = np.unique(sig['deg_pool'].astype(int), return_counts=True)
    muo_deg_vals, muo_deg_cnts = np.unique(muo['deg_pool'].astype(int), return_counts=True)
    ax.loglog(sig_deg_vals, sig_deg_cnts, 'o-', ms=3, label='Signal', alpha=0.7)
    ax.loglog(muo_deg_vals, muo_deg_cnts, 'o-', ms=3, label='Muon', alpha=0.7)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Degree distribution (log-log)')
    fig.tight_layout()
    save(fig, '25_degree_loglog.pdf')

    # ── 29. Event size distribution (stacked) ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(sig['n_nodes'].max(), muo['n_nodes'].max()) + 100, 60)
    ax.hist([sig['n_nodes'], muo['n_nodes']], bins=bins, stacked=True,
            label=['Signal', 'Muon'], alpha=0.7, color=['C0', 'C3'])
    ax.set_xlabel('Nodes per event')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Event size distribution (stacked)')
    fig.tight_layout()
    save(fig, '26_event_size_stacked.pdf')

    # ── DONE ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  All plots saved to: {OUT_DIR}/")
    print(f"  Total: {len(sig_list) + len(muo_list)} events analyzed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
