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

def main():
    print("=" * 60)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Load raw data (old format — no edge_index)
    sig_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    bkg_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/background/30011001/*.pt'))
    
    sig_data = torch.load(sig_files[0], weights_only=False, map_location='cpu')[:4]
    bkg_data = torch.load(bkg_files[0], weights_only=False, map_location='cpu')[:4]
    
    print(f"Loaded {len(sig_data)} signal + {len(bkg_data)} background events")
    
    # 2. Build graphs on-the-fly (simulating what the DataLoader does)
    batch_list = []
    t0 = time.time()
    for d in sig_data + bkg_data:
        if not hasattr(d, 'edge_index') or d.edge_index is None:
            d.edge_index = build_velo_graph(d.pos, d.x_cat)
            d.edge_attr = compute_edge_attr(d.pos, d.x_cont, d.edge_index)
        d.num_nodes = d.x_cont.shape[0]
        batch_list.append(d)
    
    graph_time = time.time() - t0
    print(f"Graph construction for {len(batch_list)} events: {graph_time:.2f}s")
    
    # 3. Batch
    batch = Batch.from_data_list(batch_list)
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes, "
          f"{batch.edge_index.shape[1]} edges")
    
    # 4. Lightning module forward + loss
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
    
    # 5. Check gradient health
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
