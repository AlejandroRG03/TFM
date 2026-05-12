#!/usr/bin/env python
"""Test script for validating graph construction and model forward pass."""
import torch
import glob
import time
import sys

sys.path.append('/home3/alejandro.rodriguez/TFM/GNN/utils')
sys.path.append('/home3/alejandro.rodriguez/TFM/GNN')

from build_graph import build_velo_graph, compute_edge_attr
from codex_gnn_model import CODEXVetoGNN
from torch_geometric.data import Data, Batch

def test_graph_construction():
    print("=" * 60)
    print("TEST 1: Graph Construction on Real Events")
    print("=" * 60)
    
    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    if not files:
        print("ERROR: No data files found!")
        return False
    
    data_list = torch.load(files[0], weights_only=False, map_location='cpu')
    
    for i, d in enumerate(data_list[:5]):
        t0 = time.time()
        edge_index = build_velo_graph(d.pos, d.x_cat)
        edge_attr = compute_edge_attr(d.pos, d.x_cont, edge_index)
        dt = time.time() - t0
        
        n = d.pos.shape[0]
        e = edge_index.shape[1]
        print(f"  Event {i}: N={n:5d} | E={e:6d} | avg_deg={e/n:.1f} | "
              f"edge_attr={list(edge_attr.shape)} | time={dt:.3f}s")
        
        # Sanity checks
        assert edge_index.shape[0] == 2, f"Bad shape: {edge_index.shape}"
        assert edge_attr.shape == (e, 10), f"Bad edge_attr: {edge_attr.shape}"
        assert edge_index.max() < n, f"Index OOB: {edge_index.max()} >= {n}"
        assert edge_index.min() >= 0, f"Negative index"
        assert not torch.isnan(edge_attr).any(), "NaN in edge_attr"
        assert not torch.isinf(edge_attr).any(), "Inf in edge_attr"
    
    print("  >> PASSED\n")
    return True


def test_model_forward():
    print("=" * 60)
    print("TEST 2: Model Forward Pass")
    print("=" * 60)
    
    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    data_list = torch.load(files[0], weights_only=False, map_location='cpu')
    
    # Build graphs for a small batch
    batch_data = []
    for d in data_list[:4]:
        edge_index = build_velo_graph(d.pos, d.x_cat)
        edge_attr = compute_edge_attr(d.pos, d.x_cont, edge_index)
        batch_data.append(Data(
            x_cont=d.x_cont,
            x_cat=d.x_cat,
            pos=d.pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=d.y,
            global_attr=d.global_attr,
            num_nodes=d.x_cont.shape[0]
        ))
    
    batch = Batch.from_data_list(batch_data)
    print(f"  Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes, "
          f"{batch.edge_index.shape[1]} total edges")
    
    # Create model
    model = CODEXVetoGNN()
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    with torch.no_grad():
        t0 = time.time()
        out = model(batch)
        dt = time.time() - t0
    
    print(f"  Output shape: {out.shape}")
    print(f"  Output values: {out.squeeze().tolist()}")
    print(f"  Forward pass time: {dt:.3f}s (CPU)")
    
    assert out.shape == (4, 1), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    
    print("  >> PASSED\n")
    return True


def test_backward():
    print("=" * 60)
    print("TEST 3: Backward Pass (Gradient Flow)")
    print("=" * 60)
    
    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    data_list = torch.load(files[0], weights_only=False, map_location='cpu')
    
    batch_data = []
    for d in data_list[:2]:
        edge_index = build_velo_graph(d.pos, d.x_cat)
        edge_attr = compute_edge_attr(d.pos, d.x_cont, edge_index)
        batch_data.append(Data(
            x_cont=d.x_cont, x_cat=d.x_cat, pos=d.pos,
            edge_index=edge_index, edge_attr=edge_attr,
            y=d.y, global_attr=d.global_attr, num_nodes=d.x_cont.shape[0]
        ))
    
    batch = Batch.from_data_list(batch_data)
    
    model = CODEXVetoGNN()
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    out = model(batch).squeeze(-1)
    loss = criterion(out, batch.y)
    loss.backward()
    
    # Check gradients flow to all parameter groups
    groups = {
        'module_emb': model.module_emb,
        'node_encoder': model.node_encoder,
        'interaction_layers': model.layers,
        'global_pool': model.global_pool,
        'classifier': model.classifier
    }
    
    all_ok = True
    for name, module in groups.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in module.parameters() if p.requires_grad)
        status = "OK" if has_grad else "NO GRADIENT!"
        if not has_grad:
            all_ok = False
        print(f"  {name:25s}: {status}")
    
    print(f"  Loss value: {loss.item():.4f}")
    
    if all_ok:
        print("  >> PASSED (all parameter groups receive gradients)\n")
    else:
        print("  >> FAILED!\n")
    
    return all_ok


if __name__ == "__main__":
    ok1 = test_graph_construction()
    ok2 = test_model_forward()
    ok3 = test_backward()
    
    print("=" * 60)
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
