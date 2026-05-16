#!/usr/bin/env python
"""Test script for validating graph construction and model forward/backward pass."""
import torch
import glob
import time
import sys
import random

sys.path.append('/home3/alejandro.rodriguez/TFM/GNN/utils')
sys.path.append('/home3/alejandro.rodriguez/TFM/GNN')

from build_graph import build_velo_graph, compute_edge_attr, compute_batched_edge_attr
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
        edge_attr = compute_edge_attr(d.pos, d.x_cont[:, :9], edge_index)
        dt = time.time() - t0

        n = d.pos.shape[0]
        e = edge_index.shape[1]
        print(f"  Event {i}: N={n:5d} | E={e:6d} | avg_deg={e/n:.1f} | "
              f"edge_attr={list(edge_attr.shape)} | time={dt:.3f}s")

        assert edge_index.shape[0] == 2, f"Bad shape: {edge_index.shape}"
        assert edge_attr.shape == (e, 10), f"Bad edge_attr: {edge_attr.shape}"
        assert edge_index.max() < n, f"Index OOB: {edge_index.max()} >= {n}"
        assert edge_index.min() >= 0, f"Negative index"
        assert not torch.isnan(edge_attr).any(), "NaN in edge_attr"
        assert not torch.isinf(edge_attr).any(), "Inf in edge_attr"

    print("  >> PASSED\n")
    return True


def test_model_forward_backward_compat():
    """
    Test forward and backward with existing 9-dim data (backward compat).
    Uses explicit n_cont_features=9 to match existing files.
    """
    print("=" * 60)
    print("TEST 2a: Forward + Backward (9-dim compat)")
    print("=" * 60)

    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    data_list = torch.load(files[0], weights_only=False, map_location='cpu')

    batch_data = []
    for d in data_list[:4]:
        ei = build_velo_graph(d.pos, d.x_cat)
        ea = compute_edge_attr(d.pos, d.x_cont[:, :9], ei)
        batch_data.append(Data(
            x_cont=d.x_cont[:, :9], x_cat=d.x_cat, pos=d.pos,
            edge_index=ei, edge_attr=ea,
            y=d.y, global_attr=d.global_attr, num_nodes=d.x_cont.shape[0]
        ))

    batch = Batch.from_data_list(batch_data)
    print(f"  Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes, "
          f"{batch.edge_index.shape[1]} total edges")

    model = CODEXVetoGNN(n_cont_features=9)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model(batch)
        dt = time.time() - t0

    print(f"  Forward: {out.shape} | values={out.squeeze().tolist()} | time={dt:.3f}s")
    assert out.shape == (4, 1), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    # Backward
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    out = model(batch).squeeze(-1)
    loss = criterion(out, batch.y)
    loss.backward()

    groups = {
        'module_emb': model.module_emb,
        'node_encoder': model.node_encoder,
        'interaction_layers': model.layers,
        'attn_pool': model.attn_pool,

        'classifier': model.classifier,
    }
    all_ok = True
    for name, module in groups.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in module.parameters() if p.requires_grad)
        status = "OK" if has_grad else "NO GRADIENT!"
        if not has_grad:
            all_ok = False
        print(f"  {name:25s}: {status}")

    print(f"  Loss: {loss.item():.4f}")
    print("  >> PASSED\n" if all_ok else "  >> FAILED!\n")
    return all_ok


def test_model_11_features():
    """
    Test forward+backward with synthetic 11-dim data, matching the new
    default n_cont_features=11 (after feature engineering).
    """
    print("=" * 60)
    print("TEST 2b: Forward + Backward (11-dim, synthetic)")
    print("=" * 60)

    random.seed(42)
    torch.manual_seed(42)

    graphs = []
    for gid in range(4):
        n = random.randint(50, 200)
        x_cont = torch.randn(n, 11)
        x_cat = torch.randint(0, 52, (n,))
        pos = torch.randn(n, 3) * 50
        global_attr = torch.randn(1, 3)
        y = torch.tensor([random.randint(0, 1)], dtype=torch.float)

        ei = build_velo_graph(pos, x_cat)
        if ei.shape[1] == 0:
            continue
        ea = compute_edge_attr(pos, x_cont, ei)

        graphs.append(Data(
            x_cont=x_cont, x_cat=x_cat, pos=pos,
            edge_index=ei, edge_attr=ea,
            y=y, global_attr=global_attr, num_nodes=n
        ))

    if not graphs:
        print("  >> SKIPPED (no graphs built from synthetic data)\n")
        return True

    batch = Batch.from_data_list(graphs)
    print(f"  Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes, "
          f"{batch.edge_index.shape[1]} total edges")

    model = CODEXVetoGNN()  # uses default n_cont_features=11
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (len(graphs), 1), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"  Forward: {out.shape} | values={out.squeeze().tolist()}")

    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    out = model(batch).squeeze(-1)
    loss = criterion(out, batch.y)
    loss.backward()

    groups = {
        'module_emb': model.module_emb,
        'node_encoder': model.node_encoder,
        'interaction_layers': model.layers,
        'attn_pool': model.attn_pool,

        'classifier': model.classifier,
    }
    all_ok = True
    for name, module in groups.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in module.parameters() if p.requires_grad)
        status = "OK" if has_grad else "NO GRADIENT!"
        if not has_grad:
            all_ok = False
        print(f"  {name:25s}: {status}")

    print(f"  Loss: {loss.item():.4f}")
    print("  >> PASSED\n" if all_ok else "  >> FAILED!\n")
    return all_ok


def test_batched_edge_attr():
    """
    Test that compute_batched_edge_attr produces the same output
    as calling compute_edge_attr sequentially per graph.
    """
    print("=" * 60)
    print("TEST 3: Batched Edge Attr vs Sequential")
    print("=" * 60)

    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    data_list = torch.load(files[0], weights_only=False, map_location='cpu')

    # Prepare graphs sequentially
    pos_list, x_cont_list, ei_list = [], [], []
    ref_attrs = []
    for d in data_list[:10]:
        ei = build_velo_graph(d.pos, d.x_cat)
        if ei.shape[1] == 0:
            continue
        ei = ei.to(torch.long)
        ea = compute_edge_attr(d.pos, d.x_cont[:, :9], ei)
        pos_list.append(d.pos)
        x_cont_list.append(d.x_cont[:, :9])
        ei_list.append(ei)
        ref_attrs.append(ea)

    if not pos_list:
        print("  >> SKIPPED (no valid events)\n")
        return True

    # Batched computation
    batched_attrs = compute_batched_edge_attr(pos_list, x_cont_list, ei_list)

    # Compare
    all_close = True
    for i, (ref, bat) in enumerate(zip(ref_attrs, batched_attrs)):
        if not torch.allclose(ref, bat, atol=1e-6):
            print(f"  Event {i}: MISMATCH! max_diff={ (ref-bat).abs().max().item():.2e }")
            all_close = False
        else:
            print(f"  Event {i}: OK  ({ref.shape[0]:6d} edges)")

    if all_close:
        print("  >> PASSED (batched == sequential)\n")
    else:
        print("  >> FAILED!\n")
    return all_close


if __name__ == "__main__":
    ok1 = test_graph_construction()
    ok2 = test_model_forward_backward_compat()
    ok3 = test_model_11_features()
    ok4 = test_batched_edge_attr()

    print("=" * 60)
    if ok1 and ok2 and ok3 and ok4:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
