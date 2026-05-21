#!/usr/bin/env python
"""
End-to-end integration test: loads real data from disk, builds graphs
on-the-fly (like the DataLoader would), and runs through the full
Lightning module (forward + loss + backward).
"""
import sys
sys.path.append('/home3/alejandro.rodriguez/TFM/GNN')
sys.path.append('/home3/alejandro.rodriguez/TFM/GNN/utils')

import torch
import glob
import time
from utils.build_graph import build_velo_graph, compute_edge_attr
from codex_gnn_model import CODEXVetoGNN
from lightning_model import CODEXLightning
from torch_geometric.data import Data, Batch


def strip_eta(x_cont):
    if x_cont.shape[1] in (9, 11):
        return x_cont[:, [0, 1, 2, 3, 4, 6, 7, 8]].contiguous()
    return x_cont


def load_events(filepath, n=5):
    raw = torch.load(filepath, weights_only=False, map_location='cpu')
    if isinstance(raw, dict):
        keys = [k for k in raw if k.startswith('data.')]
        slice_keys = [k for k in raw if k.startswith('slices.')]
        n_events = raw[slice_keys[0]].shape[0] - 1 if slice_keys else 1
        events = []
        for i in range(min(n_events, n)):
            d = Data()
            for k in keys:
                attr = k.split('.')[1]
                slice_key = 'slices.' + attr
                if slice_key in raw:
                    start, end = raw[slice_key][i], raw[slice_key][i + 1]
                    if attr == 'edge_index':
                        setattr(d, attr, raw[k][:, start:end].contiguous())
                    else:
                        setattr(d, attr, raw[k][start:end])
                else:
                    setattr(d, attr, raw[k][i])
            events.append(d)
        return events
    return list(raw)[:n]


def main():
    print("=" * 60)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 60)

    sig_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    bkg_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/background/30011001/*.pt'))

    sig_data = load_events(sig_files[0], n=4)
    bkg_data = load_events(bkg_files[0], n=4)

    print(f"Loaded {len(sig_data)} signal + {len(bkg_data)} background events")

    batch_list = []
    t0 = time.time()
    for d in sig_data + bkg_data:
        x_cont = strip_eta(d.x_cont)
        if not hasattr(d, 'edge_index') or d.edge_index is None:
            module_ids = d.x_cat if hasattr(d, 'x_cat') else torch.zeros(d.pos.shape[0], dtype=torch.long)
            d.edge_index = build_velo_graph(d.pos, module_ids)
            d.edge_attr = compute_edge_attr(d.pos, x_cont, d.edge_index)
        # Skip events with no edges (isolated hits)
        if d.edge_index.shape[1] == 0:
            continue
        d.edge_index = d.edge_index.to(torch.long)
        d.x_cont = x_cont
        d.num_nodes = x_cont.shape[0]
        if hasattr(d, 'x_cat'):
            del d.x_cat
        batch_list.append(d)

    graph_time = time.time() - t0
    print(f"Graph construction for {len(batch_list)} events: {graph_time:.2f}s")

    batch = Batch.from_data_list(batch_list)
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes, "
          f"{batch.edge_index.shape[1]} edges")

    model = CODEXLightning(pos_weight_val=1.0, learning_rate=5e-4)
    model.train()

    t0 = time.time()
    logits = model(batch)
    loss = model.criterion(logits, batch.y.float())
    loss.backward()
    fwd_time = time.time() - t0

    print(f"\nForward + backward: {fwd_time:.3f}s")
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits: {logits.detach().squeeze().tolist()}")
    print(f"Labels: {batch.y.tolist()}")

    total_grad_norm = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
            n_params += 1
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\nGrad norm: {total_grad_norm:.4f} (across {n_params} parameter tensors)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
