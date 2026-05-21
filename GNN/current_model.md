# Technical Report: Architecture, Physical Motivation and Design of the CODEX-b GNN System

This document provides a rigorous scientific and technical exposition of the complete veto system for CODEX-b. It covers everything from data extraction from the LHCb VELO subdetector to the training pipeline, detailing each design decision and its physical justification.

---

## 1. Data Preparation Pipeline

### 1.1. ROOT Extraction (`prepare_torch_files.py`)

Raw data resides in ROOT files (`Ntuple VeloMultiTuple`) at `/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/`. The pipeline:

1. **Chunked reading**: `uproot.iterate()` reads the tree in ~100 MB blocks to avoid RAM saturation.
2. **Feature engineering**: `r_T`, `phi` (cylindrical coordinates) and `codex_angle` (projective angle toward CODEX-b) are computed via functions in `python_modules`.
3. **Normalisation**: Continuous variables are normalised using precomputed means and standard deviations stored in `stats/global_normalization_stats.json`. This ensures numerical stability during training.
4. **Graph construction**: Each event is processed individually via `build_velo_graph()` (see Section 2).
5. **Repacked format storage**: Graphs are collated into a dictionary with contiguous tensors (`data.x_cont`, `data.edge_index`, etc.) + `slices` for indexing. This enables efficient mmap-based loading.

**Dataset variables:**

| Type | Variables | Dimensionality |
|---|---|---|
| Continuous (node) | `x`, `y`, `z`, `r_T`, `phi`, `n_pix`, `codex_angle`, `module_side` | 8 |
| Raw position | `x_raw`, `y_raw`, `z_raw` (mm) | 3 |
| Global (event) | `nVtx_per_event`, `nClu_per_event`, `nTrk_per_event` | 3 |
| Edge | Spatial differences + distance + cylindrical deltas + direction vector | 10 |

**z-coverage**: Hits with `z >= -150` mm are kept (active VELO region).

---

## 2. Physically Motivated Graph Construction

`utils/build_graph.py` implements a static graph that exploits the known topology of the VELO detector. The graph is built once during data preparation (CPU), eliminating the need for dynamic construction during training.

### 2.1. Graph Topology

**Intra-module edges (local clustering)**:
- `radius_graph` in the xy sensor plane with `r < 5.0 mm`.
- Connects hits within the same module, allowing the network to identify shared charge deposits or secondary electrons (delta rays).

**Inter-module edges (KNN to M±1)**:
- KNN (`k=3`) from module `M_i` to `M_{i±1}` in the xy plane.
- Tracks traverse silicon planes sequentially; connecting hits in adjacent layers preselects congruent track segments.

**Skip edges (KNN to M±2)**:
- KNN (`k=1`) from module `M_i` to `M_{i±2}`.
- Provides robustness against sensor inefficiencies or dead regions, allowing information to "skip" a plane.

**Bidirectionality**: All edges are duplicated (reversed direction) and duplicates are removed via coalescence.

### 2.2. Edge Attributes (10-dimensional)

`compute_edge_attr()` generates rich geometric features for each edge:

| Component | Description |
|---|---|
| `dx, dy, dz` | Raw spatial differences (mm) |
| `dist_3d` | 3D Euclidean distance |
| `d_rT, d_phi, d_z_n` | Cylindrical deltas (normalised) |
| `ux, uy, uz` | 3D unit direction vector |

---

## 3. Model: Interaction Network (V4.1)

`codex_gnn_model.py` implements an **Interaction Network (V4.1)** optimised for assimilating geometric constraints via message passing.

### 3.1. General Architecture

```
Node Features (8) → Node Encoder → 4× InteractionLayer → Multi-head Pooling → Classifier
Edge Features (10) → Edge Encoder ────────────────────────────┘
```

### 3.2. Changes from V3 (/V2)

| Decision | V2 | V3 | V4.1 (current) |
|---|---|---|---|
| Attention mechanism | GATv2Conv | InteractionNetwork | InteractionNetwork |
| Module embedding | Yes | Yes | Removed (not useful for discrimination) |
| Continuous features | 9 | 9 | **8** (`eta` dropped, r=0.77 with `codex_angle`) |
| Dynamic graph | KNN in forward | Static (precomputed) | Static (precomputed) |
| `num_layers` (training) | — | 5 | **4** (config: 5 in lightning_train.py) |
| `hidden_channels` | — | 128 | **128** |
| `edge_hidden` | — | 96 | **96** |

### 3.3. InteractionLayer (Message Passing)

Pure Interaction Network layer (no attention). Based on PyTorch Geometric's `MessagePassing` with `add` aggregation.

**Message function** (Edge MLP):
```
m_{ij} = edge_mlp( [x_i ‖ x_j ‖ edge_attr_{ij}] )
```
Architecture: `Linear(2·node_dim + edge_dim, 64) → LayerNorm → SiLU → Linear(64, node_dim)`. Narrow bottleneck because it is applied **per edge** (~millions per batch).

**Update function** (Node MLP):
```
x_i' = node_mlp( [x_i ‖ Σ_j m_{ij}] ) + x_i  # residual
```
Architecture: `Linear(2·node_dim, node_dim) → SiLU → Linear(node_dim, node_dim)`. Wider because it is applied **per node**.

**Regularisation**: LayerNorm in residual + inside each MLP.

### 3.4. Initial Encoders

**Node Encoder**: `Linear(8, 128) → LayerNorm → SiLU`. Projects the 8 continuous features into hidden space.

**Edge Encoder**: `Linear(10, 32) → LayerNorm → SiLU`. Normalises edge features (mm-scale → stable) and compresses to 32D. Prevents fp16/bf16 overflow.

### 3.5. Global Pooling (Jumping Knowledge)

Features are extracted from **two levels** of the InteractionLayer stack:
- **Middle layer** (`jk_mid_layer = num_layers // 2 - 1`): local track fragments.
- **Final layer** (`num_layers - 1`): global graph structure.

Two pooling strategies are applied at each level:
- **AttentionalAggregation**: Learns importance weights for each node via a `gate_nn` (MLP: `Linear(128, 64) → SiLU → Linear(64, 1)`).
- **GlobalMaxPool**: Captures the most extreme or energetic hits.

### 3.6. Classifier

3-layer MLP with Dropout:

```
pool_dim = (128 × 2) × 2 + 3 = 515  # (attn+max) × 2 levels + 3 global features
Linear(515, 256) → SiLU → Dropout(0.3) → Linear(256, 128) → SiLU → Dropout(0.3) → Linear(128, 1)
```

---

## 4. Training Pipeline (PyTorch Lightning)

### 4.1. LightningModule (`lightning_model.py`)

**Optimiser**: AdamW with `weight_decay=1e-4` (L2 regularisation).

**LR Scheduler**: `WarmupReduceLROnPlateau` — linear warmup for 2 epochs (from `1e-6` to base LR) followed by `ReduceLROnPlateau(factor=0.5, patience=3)` monitoring `val_loss`.

**Loss**: `BCEWithLogitsLoss` with `pos_weight=1.0` (1:1 balance via paired sampling).

**Metrics**: BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision (note: `batch.y` is cast to `long()` since torchmetrics requires integer labels for PR curve computation).

**Mixed precision**: BF16 to avoid overflows in squared spatial differences.

### 4.2. Training Configuration (`lightning_train.py`)

| Parameter | Value | Justification |
|---|---|---|
| `BATCH_SIZE` | 32 | Avoids CUDA OOM (~16k edges/graph × 32 = 2.1M edges/batch) |
| `ACCUM_STEPS` | 4 | Effective batch size of 128 |
| `NUM_WORKERS` | 1 | Prevents OST contention on Lustre |
| `prefetch_factor` | 8 | GPU buffer of ~512 graphs (~5s) |
| `MP_CONTEXT` | spawn | Avoids deadlocks with mmap on Lustre |
| `USE_MULTI_GPU` | False | DDP causes NCCL timeouts with slow I/O |
| `gradient_clip_val` | 1.0 | Stabilises gradients in deep networks |

### 4.3. Data Loading and Anti-Bottleneck I/O Strategy

**Repacked format**: Chunks (~6 GB with ~5,700 graphs) are stored as contiguous tensors + slices. `_load_repacked_chunk()` performs a **sequential read** (`mmap=False`) to avoid the thousands of page faults that occurred with `mmap=True` on Lustre OSTs at 100% capacity.

**Concurrent signal || background loading**: Two threads (`threading.Thread`) load signal (OST:1) and background (OST:0) simultaneously, exploiting their placement on different OSTs with `stripe_count=1`. This reduces per-pair load time from ~219s to ~95-110s.

**Triple-queue double-buffer**: Inside the DataLoader worker, a `daemon=True` thread pre-loads subsequent chunks into a `queue.Queue(maxsize=3)` while the main thread yields graphs. The 3-slot queue provides ~196s of GPU time cushion, absorbing Lustre latency spikes.

**Progressive expansion**: Training starts with 20 pairs and expands to 70 then all available according to the schedule `[(2, 20), (7, 70), (999999, None)]`. This stabilises initial weights before exposing the model to full data variability.

### 4.4. Early Stopping and Checkpointing

- `EarlyStopping(patience=10, monitor="val_loss")` stops training if no improvement.
- `ModelCheckpoint(save_top_k=1, monitor="val_loss")` saves the best model.
- `LearningRateMonitor` logs the learning rate to WandB.
- `WandbLogger` with `log_model="all"` for full traceability.

---

## 5. Performance Considerations

| Issue | Problem | Solution |
|---|---|---|
| Lustre OSTs at 100% | Catastrophic page faults with mmap | Sequential read (`mmap=False`) |
| OST contention | 2 workers saturated OSTs | Reduce to `NUM_WORKERS=1` |
| Load latency > compute | GPU underutilisation | Double-buffer + concurrent sig\|\|bkg loading |
| torchmetrics failure | `BinaryAveragePrecision` rejects float | Cast `batch.y.long()` |
| DDP with slow I/O | NCCL timeouts from desynchronisation | Single GPU (`USE_MULTI_GPU=False`) |

---

*This design integrates particle physics principles (track linearity in VELO) with graph architectures (Interaction Networks) and a data pipeline optimised for parallel filesystems (Lustre), resulting in a robust, scalable, and physically motivated veto system for CODEX-b.*
