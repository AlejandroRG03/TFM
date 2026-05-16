"""
Module-Aware Graph Construction for the VELO detector.

Builds a physically-motivated static graph that exploits the known geometry
of the VELO:
  - Intra-module edges: radius_graph within each module's sensor plane (xy)
  - Inter-module edges: KNN from module M_i to adjacent modules M_{i±1}
  - Skip edges:         KNN from module M_i to modules M_{i±2} (high-pT tracks)

The graph and its geometric edge attributes are computed once on the CPU
during data preparation, eliminating the need for dynamic (and non-differentiable)
graph construction during training.
"""

import torch
import numpy as np
from torch_geometric.nn import radius_graph
from torch_geometric.utils import coalesce



def build_velo_graph(pos, module_ids,
                     intra_radius=5.0,
                     inter_k=3,
                     skip_k=1,
                     max_inter_dist=15.0):
    """
    Constructs a physically-motivated graph for the VELO detector.

    Args:
        pos:             (N, 3) tensor — raw physical coordinates (x, y, z) in mm.
        module_ids:      (N,)   tensor — integer module ID for each hit.
        intra_radius:    float  — radius (mm) for intra-module edges in the xy plane.
        inter_k:         int    — number of nearest neighbours toward adjacent modules.
        skip_k:          int    — number of nearest neighbours toward M_{i±2}.
        max_inter_dist:  float  — maximum xy-distance (mm) allowed for inter-module edges.

    Returns:
        edge_index: (2, E) LongTensor of bidirectional edges (no self-loops).
    """
    all_edges = []
    unique_modules = module_ids.unique().sort()[0]

    # Pre-compute per-module masks and indices for efficiency
    module_data = {}
    for mod_id in unique_modules:
        mask = (module_ids == mod_id)
        idx = mask.nonzero(as_tuple=True)[0]
        module_data[mod_id.item()] = (idx, pos[idx])

    for mod_id_val in unique_modules.tolist():
        idx_i, pos_i = module_data[mod_id_val]
        n_i = idx_i.shape[0]

        # ── INTRA-MODULE: radius graph in the xy sensor plane ──────────
        if n_i > 1:
            # Use only xy for intra-module connectivity (hits on the same sensor plane)
            local_edges = radius_graph(pos_i[:, :2], r=intra_radius, loop=False)
            all_edges.append(idx_i[local_edges])

        # ── INTER-MODULE: connect to adjacent modules (M ± 1) ─────────
        for delta in (1, -1):
            adj_mod = mod_id_val + delta
            if adj_mod not in module_data:
                continue
            idx_j, pos_j = module_data[adj_mod]
            _add_cross_module_edges(all_edges, idx_i, pos_i, idx_j, pos_j,
                                    inter_k, max_inter_dist)

        # ── SKIP CONNECTIONS: modules M ± 2 (high-pT tracks) ──────────
        for delta in (2, -2):
            skip_mod = mod_id_val + delta
            if skip_mod not in module_data:
                continue
            idx_s, pos_s = module_data[skip_mod]
            _add_cross_module_edges(all_edges, idx_i, pos_i, idx_s, pos_s,
                                    skip_k, max_inter_dist * 1.2)

    if all_edges:
        edge_index = torch.cat(all_edges, dim=1)
        # Make bidirectional, remove duplicates and sort
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = coalesce(edge_index, num_nodes=pos.size(0))

    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return edge_index


def _add_cross_module_edges(all_edges, idx_i, pos_i, idx_j, pos_j, k, max_dist):
    """Helper: KNN from module i → module j in the xy plane, with distance cut."""
    n_j = idx_j.shape[0]
    k_eff = min(k, n_j)
    if k_eff == 0:
        return

    ni, nj = pos_i.shape[0], pos_j.shape[0]

    # Pairwise distances in xy only (ignore z for cross-module — z separation is fixed)
    if ni * nj > 2_000_000:
        # Chunked cdist to keep memory bounded
        chunk_size = 500
        all_src_list, all_dst_list = [], []
        for start in range(0, ni, chunk_size):
            end = min(start + chunk_size, ni)
            dists = torch.cdist(pos_i[start:end, :2], pos_j[:, :2])
            topk_dists, topk_local = dists.topk(k_eff, dim=1, largest=False)
            valid = topk_dists < max_dist
            src_local, dst_col = valid.nonzero(as_tuple=True)
            if src_local.numel() > 0:
                dst_local = topk_local[src_local, dst_col]
                all_src_list.append(idx_i[start + src_local])
                all_dst_list.append(idx_j[dst_local])
        if all_src_list:
            all_edges.append(torch.stack([torch.cat(all_src_list), torch.cat(all_dst_list)], dim=0))
        return

    dists = torch.cdist(pos_i[:, :2], pos_j[:, :2])        # (N_i, N_j)
    topk_dists, topk_local = dists.topk(k_eff, dim=1, largest=False)

    # Apply distance threshold
    valid_mask = topk_dists < max_dist
    src_local, dst_col = valid_mask.nonzero(as_tuple=True)
    dst_local = topk_local[src_local, dst_col]

    if src_local.numel() > 0:
        src_global = idx_i[src_local]
        dst_global = idx_j[dst_local]
        all_edges.append(torch.stack([src_global, dst_global], dim=0))


# _coalesce_edges removed in favour of torch_geometric.utils.coalesce



def compute_edge_attr(pos_raw, x_cont, edge_index):
    """
    Computes rich geometric edge attributes from raw physical coordinates
    and normalised continuous features.

    The resulting edge features encode:
      - Δx, Δy, Δz (raw mm differences — directional information)
      - Euclidean distance in 3D
      - Δr_T, Δφ, Δz_norm (cylindrical coordinate differences, normalised)
      - Normalised unit direction vector (3D)

    Args:
        pos_raw:    (N, 3) tensor — raw coordinates (x, y, z) in mm.
        x_cont:     (N, 9) tensor — normalised continuous features
                    [x, y, z, r_T, phi, eta, n_pix, codex_angle, module_side].
        edge_index: (2, E) LongTensor.

    Returns:
        edge_attr:  (E, 10) tensor of geometric edge features.
    """
    src, dst = edge_index[0], edge_index[1]

    # 1. Raw spatial differences (mm) — encodes direction + scale
    dx = pos_raw[src, 0] - pos_raw[dst, 0]                     # (E,)
    dy = pos_raw[src, 1] - pos_raw[dst, 1]
    dz = pos_raw[src, 2] - pos_raw[dst, 2]
    dist_3d = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)         # (E,)

    # 2. Cylindrical deltas from normalised features
    #    x_cont columns: [x, y, z, r_T, phi, eta, n_pix, codex_angle, module_side]
    #    Indices:          0  1  2   3    4    5     6        7            8
    d_rT   = x_cont[src, 3] - x_cont[dst, 3]                   # Δr_T (normalised)
    d_phi  = x_cont[src, 4] - x_cont[dst, 4]                   # Δφ   (normalised)
    d_z_n  = x_cont[src, 2] - x_cont[dst, 2]                   # Δz   (normalised)

    # 3. Unit direction vector (3D)
    ux = dx / dist_3d
    uy = dy / dist_3d
    uz = dz / dist_3d

    edge_attr = torch.stack([
        dx * 0.1, dy * 0.1, dz * 0.1, # raw spatial differences (scaled to ~cm)
        dist_3d * 0.1,                # Euclidean distance      (scaled to ~cm)
        d_rT, d_phi, d_z_n,           # cylindrical deltas      (already normalised)
        ux, uy, uz                    # unit direction          (normalised)
    ], dim=-1)               # Total: 10

    return edge_attr



def compute_batched_edge_attr(pos_list, x_cont_list, edge_index_list):
    """
    Compute edge_attr for multiple graphs in one batched operation.

    Concatenates all nodes and edges across graphs, computes edge_attr
    on the merged tensors, then splits back by graph. This replaces N
    small Python-level calls with 1 vectorised call, reducing Python
    loop overhead and improving CPU cache utilisation.

    Args:
        pos_list:        list of (N_i, 3) tensors — raw coordinates.
        x_cont_list:     list of (N_i, C) tensors — normalised features.
        edge_index_list: list of (2, E_i) tensors — (int32 or int64).

    Returns:
        list of (E_i, 10) tensors — one edge_attr per graph.
    """
    # 1. Compute node offsets per graph for edge index re-indexing
    offsets = [0]
    for p in pos_list:
        offsets.append(offsets[-1] + p.shape[0])

    # 2. Concatenate all nodes
    all_pos = torch.cat(pos_list, dim=0)
    all_x_cont = torch.cat(x_cont_list, dim=0)

    # 3. Offset edge indices so they point into the global node array,
    #    then concatenate all edges
    all_edge_parts = []
    for ei, offset in zip(edge_index_list, offsets[:-1]):
        all_edge_parts.append(ei + offset)
    all_edge_index = torch.cat(all_edge_parts, dim=1)

    # 4. Single batched computation
    all_edge_attr = compute_edge_attr(all_pos, all_x_cont, all_edge_index)

    # 5. Split back by graph using edge counts
    edge_counts = [ei.shape[1] for ei in edge_index_list]
    cumsum = [0]
    for c in edge_counts:
        cumsum.append(cumsum[-1] + c)

    return [all_edge_attr[cs:ce] for cs, ce in zip(cumsum[:-1], cumsum[1:])]
