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


def load_events(filepath, n=5):
    """Load chunk and return list of Data objects (handles both dict and list formats)."""
    raw = torch.load(filepath, weights_only=False, map_location='cpu')
    if isinstance(raw, dict):
        keys = [k for k in raw if k.startswith('data.')]
        slice_keys = [k for k in raw if k.startswith('slices.')]
        n_events = raw[slice_keys[0]].shape[0] - 1 if slice_keys else len(raw[keys[0]])
        events = []
        for i in range(min(n_events, n)):
            d = Data()
            for k in keys:
                attr = k.split('.')[1]
                slice_key = 'slices.' + attr
                if slice_key in raw:
                    start, end = raw[slice_key][i], raw[slice_key][i + 1]
                    # edge_index has shape [2, E] — slice along dim=1, not dim=0
                    if attr == 'edge_index':
                        setattr(d, attr, raw[k][:, start:end].contiguous())
                    else:
                        setattr(d, attr, raw[k][start:end])
                else:
                    setattr(d, attr, raw[k][i])
            events.append(d)
        return events
    else:
        return list(raw)[:n]


def strip_eta(x_cont):
    """Remove eta (column 5) from x_cont. Handle 9 or 11-column input."""
    if x_cont.shape[1] in (9, 11):
        return x_cont[:, [0, 1, 2, 3, 4, 6, 7, 8]].contiguous()
    return x_cont


def test_graph_construction():
    print("=" * 60)
    print("TEST 1: Graph Construction on Real Events")
    print("=" * 60)

    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    if not files:
        print("ERROR: No data files found!")
        return False

    events = load_events(files[0], n=5)

    for i, d in enumerate(events):
        t0 = time.time()
        module_ids = d.x_cat if hasattr(d, 'x_cat') and d.x_cat is not None else torch.zeros(d.pos.shape[0], dtype=torch.long)
        edge_index = build_velo_graph(d.pos, module_ids)
        edge_attr = compute_edge_attr(d.pos, strip_eta(d.x_cont), edge_index)
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


def test_model_forward():
    print("=" * 60)
    print("TEST 2: Model Forward Pass")
    print("=" * 60)

    files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    events = load_events(files[0], n=4)

    batch_data = []
    for d in events:
        x_cont = strip_eta(d.x_cont)
        module_ids = d.x_cat if hasattr(d, 'x_cat') else torch.zeros(d.pos.shape[0], dtype=torch.long)
        edge_index = build_velo_graph(d.pos, module_ids)
        edge_attr = compute_edge_attr(d.pos, x_cont, edge_index)
        batch_data.append(Data(
            x_cont=x_cont,
            pos=d.pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=d.y,
            global_attr=d.global_attr,
            num_nodes=x_cont.shape[0]
        ))

    batch = Batch.from_data_list(batch_data)
    print(f"  Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes, "
          f"{batch.edge_index.shape[1]} total edges")

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
    events = load_events(files[0], n=2)

    batch_data = []
    for d in events:
        x_cont = strip_eta(d.x_cont)
        module_ids = d.x_cat if hasattr(d, 'x_cat') else torch.zeros(d.pos.shape[0], dtype=torch.long)
        edge_index = build_velo_graph(d.pos, module_ids)
        edge_attr = compute_edge_attr(d.pos, x_cont, edge_index)
        batch_data.append(Data(
            x_cont=x_cont, pos=d.pos,
            edge_index=edge_index, edge_attr=edge_attr,
            y=d.y, global_attr=d.global_attr, num_nodes=x_cont.shape[0]
        ))

    batch = Batch.from_data_list(batch_data)

    model = CODEXVetoGNN()
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    out = model(batch).squeeze(-1)
    loss = criterion(out, batch.y)
    loss.backward()

    groups = {
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
