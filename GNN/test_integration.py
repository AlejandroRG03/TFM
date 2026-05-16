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
from build_graph import build_velo_graph, compute_edge_attr, compute_batched_edge_attr
from codex_gnn_model import CODEXVetoGNN
from lightning_model import CODEXLightning
from torch_geometric.data import Data, Batch


def test_classic_pipeline():
    """Load old 9-dim data, build graphs per-event, run Lightning module."""
    print("=" * 60)
    print("TEST A: Classic pipeline (9-dim data, batched edge_attr)")
    print("=" * 60)

    sig_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))
    bkg_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/background/30011001/*.pt'))

    sig_data = torch.load(sig_files[0], weights_only=False, map_location='cpu')[:4]
    bkg_data = torch.load(bkg_files[0], weights_only=False, map_location='cpu')[:4]

    print(f"Loaded {len(sig_data)} signal + {len(bkg_data)} background events")

    # Prepare graph data (simulate what ChunkIterableDataset does)
    batch_list = []
    for d in sig_data + bkg_data:
        if not hasattr(d, 'edge_index') or d.edge_index is None:
            d.edge_index = build_velo_graph(d.pos, d.x_cat)
        d.edge_index = d.edge_index.to(torch.long)
        d.num_nodes = d.x_cont.shape[0]
        batch_list.append(d)

    # Batched edge_attr
    pos_list = [d.pos for d in batch_list]
    xc_list  = [d.x_cont[:, :9] for d in batch_list]
    ei_list  = [d.edge_index for d in batch_list]
    t0 = time.time()
    ea_list = compute_batched_edge_attr(pos_list, xc_list, ei_list)
    batched_time = time.time() - t0
    for d, ea in zip(batch_list, ea_list):
        d.edge_attr = ea

    # Sequential for comparison
    t0 = time.time()
    for d in batch_list:
        compute_edge_attr(d.pos, d.x_cont[:, :9], d.edge_index)
    seq_time = time.time() - t0
    print(f"  Batched edge_attr:  {batched_time:.4f}s")
    print(f"  Sequential edge_attr: {seq_time:.4f}s")
    print(f"  Speedup: {seq_time / max(batched_time, 1e-8):.1f}x")

    batch = Batch.from_data_list(batch_list)
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes, "
          f"{batch.edge_index.shape[1]} edges")

    # Lightning module (auto-detect feature dim)
    n_feats = batch_list[0].x_cont.shape[-1]
    model = CODEXLightning(
        pos_weight_val=1.0,
        learning_rate=5e-4,
        model_kwargs={"n_cont_features": n_feats}
    )
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
    print(f"Grad norm: {total_grad_norm:.4f} ({n_params} tensors)")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    assert loss.item() > 0, "Loss should be positive!"
    assert total_grad_norm > 0, "Zero gradients!"

    print("\n  >> PASSED\n")
    return True


def test_data_pipeline():
    """
    Test the ChunkIterableDataset + ProgressiveExpansionCallback
    with a tiny subset of data (2 chunk pairs).
    """
    print("=" * 60)
    print("TEST B: Data pipeline mini-integration")
    print("=" * 60)

    # Monkey-patch: import and create a minimal pipeline
    sys.path.insert(0, '/home3/alejandro.rodriguez/TFM/GNN')
    from lightning_train import ChunkIterableDataset, get_files, get_paired_files

    sig_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt'))[:3]
    bkg_files = sorted(glob.glob('/lustre/LHCb/alejandro.rodriguez/torch_data/background/30011001/*.pt'))[:3]

    pairs = get_paired_files(sig_files[:2], bkg_files[:2])
    val_pairs = get_paired_files(sig_files[2:], bkg_files[2:])

    print(f"Train pairs: {len(pairs)}, Val pairs: {len(val_pairs)}")

    # Test that set_epoch and expansion work
    ds = ChunkIterableDataset(pairs, expansion_schedule=[(5, 1), (999, 2)])
    # epoch 0 → epoch < 5 → use 1 pair
    assert ds._num_active_pairs() == 1, f"Epoch 0 should give 1 pair, got {ds._num_active_pairs()}"

    ds.set_epoch(5)
    assert ds._num_active_pairs() == 2, f"Epoch 5 should give 2 pairs, got {ds._num_active_pairs()}"

    ds.set_epoch(100)
    assert ds._num_active_pairs() == 2, f"Epoch 100 should give 2 pairs, got {ds._num_active_pairs()}"

    print("  Progressive expansion: OK")

    # Quick iteration test — load one batch from the dataset
    from torch_geometric.loader import DataLoader
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=16, num_workers=2)

    batch = None
    for i, b in enumerate(loader):
        batch = b
        if i >= 1:
            break

    if batch is not None:
        print(f"Loaded batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes")
        print(f"  x_cont shape: {batch.x_cont.shape}")
        print(f"  edge_attr shape: {batch.edge_attr.shape}")
        assert batch.x_cont.shape[-1] in (9, 11), f"Unexpected x_cont dim: {batch.x_cont.shape[-1]}"
        assert batch.edge_attr.shape[-1] == 10, f"Unexpected edge_attr dim: {batch.edge_attr.shape[-1]}"
    else:
        print("  WARNING: No batches yielded from test loader (maybe all data skipped?)")

    print("\n  >> PASSED\n")
    return True


if __name__ == "__main__":
    ok_a = test_classic_pipeline()
    ok_b = test_data_pipeline()

    print("=" * 60)
    if ok_a and ok_b:
        print("ALL INTEGRATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
