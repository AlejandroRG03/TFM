# Technical Report: Architecture, Physical Motivation and Design of the CODEX-b GNN Veto

This document provides a rigorous scientific and technical exposition of the complete veto system for CODEX-b — a GNN-based binary classifier that discriminates Standard Model background (muons, $K_L^0$) from LLP signal events using LHCb VELO hit data. It covers everything from data extraction to the training pipeline, detailing each design decision and its physical justification.

---

## 1. Data Preparation Pipeline

### 1.1. ROOT Extraction (`prepare_torch_files.py`)

Raw data resides in ROOT files (`Ntuple VeloMultiTuple`) at `/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/`. The pipeline:

1. **Chunked reading**: `uproot.iterate()` reads the tree in ~100 MB blocks to avoid RAM saturation.
2. **Beamspot centering**: `x` and `y` are shifted by `beamspotX` / `beamspotY` so coordinates are relative to the beamspot.
3. **Feature engineering**: `r_T`, `phi` (cylindrical coordinates) and `codex_angle` (projective angle toward CODEX-b) are computed via functions in `python_modules`.
4. **Normalisation**: Continuous variables are normalised using precomputed means and standard deviations stored in `stats/global_normalization_stats.json` (computed via Dask/Awkward in `obtain_stats.py`).
5. **Graph construction**: Each event is processed individually via `build_velo_graph()` (see Section 2).
6. **Repacked format storage**: Graphs are collated into a dictionary with contiguous tensors (`data.x_cont`, `data.edge_index`, etc.) + `slices` for indexing. This enables efficient loading. A `.repacked` marker file is written alongside each chunk.

**Execution**: `prepare_torch_files.py` supports `--label {MUON, KL0, SIGNAL, ALL}` and runs multiprocessing with a `ProcessPoolExecutor` (fork context, 24 workers). Events with < 3 hits are skipped. Events that produce 0 graph edges are also skipped.

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
- Tracks traverse silicon planes sequentially; connecting hits in adjacent layers preselects congruent track segments. Distance threshold of 15 mm in xy.

**Skip edges (KNN to M±2)**:
- KNN (`k=1`) from module `M_i` to `M_{i±2}`.
- Provides robustness against sensor inefficiencies or dead regions, allowing information to "skip" a plane. Distance threshold of 18 mm in xy.

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

## 3. Model: Interaction Network

`codex_gnn_model.py` implements an **Interaction Network** optimised for assimilating geometric constraints via message passing.

### 3.1. General Architecture

```
Node Features (8) → Node Encoder → 4–5× InteractionLayer → Multi-head Pooling (ALL layers) → Classifier
Edge Features (10) → Edge Encoder ────────────────────────────────────────┘
```

### 3.2. Hyperparameters

| Parameter | Model Default | Training Override |
|---|---|---|
| `n_cont_features` | 8 | auto-detected from data |
| `hidden_channels` | 96 | **128** |
| `num_layers` | 4 | **5** |
| `edge_hidden` | 128 | 128 |
| `edge_enc_dim` | 64 | 64 |
| `learning_rate` | — | **5e-4** |

### 3.3. InteractionLayer (Message Passing)

Pure Interaction Network layer (no attention). Based on PyTorch Geometric's `MessagePassing` with `add` aggregation.

**Message function** (Edge MLP):
```
m_{ij} = edge_mlp( [x_i ‖ x_j ‖ edge_attr_{ij}] )
```
Architecture: `Linear(2·node_dim + edge_dim, edge_hidden) → LayerNorm → SiLU → Linear(edge_hidden, node_dim)`. The default `edge_hidden=128` applies to millions of edges per batch.

**Update function** (Node MLP):
```
x_i' = node_mlp( [x_i ‖ Σ_j m_{ij}] ) + x_i  # residual
```
Architecture: `Linear(2·node_dim, node_dim) → SiLU → Linear(node_dim, node_dim)`. Wider because it is applied per node.

**Regularisation**: LayerNorm in residual + inside each MLP.

### 3.4. Initial Encoders

**Node Encoder**: `Linear(8, hidden_channels) → LayerNorm → SiLU`. Projects the 8 continuous features into hidden space.

**Edge Encoder**: `Linear(10, 64) → LayerNorm → SiLU`. Normalises edge features (mm-scale → stable) and projects to 64D. Prevents fp16/bf16 overflow.

### 3.5. Self-Loops

During the forward pass, self-loops are added to the edge index (each node connects to itself with a zero edge attribute vector). This allows each node to retain its own state through message passing.

### 3.6. Global Pooling (Jumping Knowledge)

Features are extracted from **every** InteractionLayer (all 4 or 5 layers), not just selected intermediate levels. This provides the classifier with a multi-resolution view of the graph representation.

Two pooling strategies are applied at each layer:
- **AttentionalAggregation**: Learns importance weights for each node via a `gate_nn` (MLP: `Linear(hidden_channels, hidden_channels//2) → SiLU → Linear(hidden_channels//2, 1)`).
- **GlobalMaxPool**: Captures the most extreme or energetic hits.

### 3.7. Classifier

Multi-layer MLP with Dropout:

```
pool_dim = (hidden_channels × 2) × num_layers + 3
         = (128 × 2) × 5 + 3 = 1283   (training config)
         =  (96 × 2) × 4 + 3 = 771    (model default)

Linear(pool_dim, hidden_channels×4) → SiLU → Dropout(0.1)
→ Linear(hidden_channels×4, hidden_channels×2) → SiLU → Dropout(0.1)
→ Linear(hidden_channels×2, 1)
```

### 3.8. Parameter Count

With training config (`hidden_channels=128, num_layers=5, edge_hidden=128`): ~1.2M parameters.

---

## 4. Training Pipeline (PyTorch Lightning)

### 4.1. LightningModule (`lightning_model.py`)

**Optimiser**: AdamW with `weight_decay=1e-4` (L2 regularisation).

**LR Scheduler**: Linear warmup for 2 epochs (from `1e-6 / lr` to 1.0) followed by `CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)`, combined via `SequentialLR`.

**Loss**: `BCEWithLogitsLoss` with `pos_weight=1.0` (1:1 balance via paired sampling). The `pos_weight` is registered as a buffer for safe device transfer.

**Metrics**: BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision (note: `batch.y` is cast to `long()` since torchmetrics requires integer labels for PR curve computation).

**Mixed precision**: BF16 to avoid overflows in squared spatial differences.

### 4.2. Training Configuration (`lightning_train.py`)

| Parameter | Value | Justification |
|---|---|---|
| `BATCH_SIZE` | 128 | H100 100 GB VRAM allows large batches (~265 MB/batch in BF16) |
| `ACCUM_STEPS` | 1 | No accumulation needed; effective batch = 128 |
| `LEARNING_RATE` | 5e-4 | — |
| `EPOCHS` | 100 | With early stopping |
| `NUM_WORKERS` | 6 | H100 has 64 cores; conservative to avoid OST contention on Lustre |
| `VAL_WORKERS` | 4 | Fewer workers for validation |
| `prefetch_factor` | 32 (train) / 8 (val) | Aggressive prefetch to keep GPU fed |
| `MP_CONTEXT` | spawn | CRITICAL: avoids deadlocks with mmap on Lustre |
| `MAX_TRAIN_PAIRS` | 45 | Limits RAM usage (~80% of available / NUM_WORKERS) |
| `MAX_VAL_PAIRS` | 5 | Limits validation to 5 chunk-pairs |
| `gradient_clip_val` | 1.0 | Stabilises gradients |
| `persistent_workers` | True | Workers stay alive across epochs, preserving their LRU chunk cache |

**Train/Val/Test split**: The last 5 chunks of each species are held out as a test set (never seen during training). The remaining chunks are split 80/20 by files (not events) to prevent data leakage. Validation is limited to `MAX_VAL_PAIRS=5` pairs.

### 4.3. Data Loading and Anti-Bottleneck I/O Strategy

**Repacked format**: Chunks (~6 GB with ~5,700 graphs) are stored as contiguous tensors + slices. `_load_repacked_chunk()` performs a **sequential read** (`mmap=False`) to avoid the thousands of page faults that occurred with `mmap=True` on Lustre OSTs at 100% capacity.

**Concurrent signal || background loading**: Two `threading.Thread` instances load signal and background chunks simultaneously via `load_chunk()`. Since `torch.load` releases the GIL during I/O, the two reads proceed in parallel, exploiting placement on different OSTs.

**Double-buffer with back-pressure**: Inside the training DataLoader worker, a `daemon=True` background thread pre-loads subsequent chunk pairs into a `queue.Queue(maxsize=2)` while the main thread yields shuffled graphs to the GPU. A 2-slot queue provides sufficient cushion to absorb Lustre latency spikes. Back-pressure uses `timeout=1.0` on `queue.put()` to remain responsive to shutdown signals.

**Per-worker LRU chunk cache**: Each worker maintains a global cache (`_CHUNK_CACHE`) that stores loaded chunks up to 80% of available RAM divided by `NUM_WORKERS`. With `persistent_workers=True`, this cache survives across epochs, dramatically reducing I/O on repeated chunks.

**Memory management**: `ctypes.CDLL("libc.so.6").malloc_trim(0)` is called after each chunk to return freed memory to the OS, preventing RSS growth.

**File validation**: Chunk pairs smaller than 100 MB are silently skipped.

### 4.4. Early Stopping and Checkpointing

- `EarlyStopping(patience=10, monitor="val_loss")` stops training if no improvement.
- `ModelCheckpoint(save_top_k=1, monitor="val_loss")` saves the best model to `models/{BKG_TYPE}/`.
- `LearningRateMonitor` logs the learning rate to WandB.
- `WandbLogger` with `log_model="all"` for full traceability.

### 4.5. Test Evaluation (`gnn_tests.py`)

After training, `gnn_tests.py` loads the best checkpoint and runs inference on the held-out test set. It computes:
- Accuracy, ROC-AUC, Precision-Recall AUC, F1 score
- Confusion matrix, classification report
- Signal efficiency vs background rejection curves
- Probability distribution plots (signal vs background)
- All plots saved to `test_plots/`.

---

## 5. Performance Considerations

| Issue | Problem | Solution |
|---|---|---|
| Lustre OSTs at 100% | Catastrophic page faults with mmap | Sequential read (`mmap=False`) |
| OST contention | Many workers saturated OSTs | Conservative `NUM_WORKERS=6` with per-worker cache |
| Load latency > compute | GPU underutilisation | Double-buffer + concurrent sig\|\|bkg loading |
| Memory fragmentation | RSS grows across epochs | `malloc_trim(0)` after each chunk |
| torchmetrics failure | `BinaryAveragePrecision` rejects float | Cast `batch.y.long()` |
| DDP with slow I/O | NCCL timeouts from desynchronisation | Single GPU (`USE_MULTI_GPU=False`) |
| Sparse edge gradients | Zero-gradient for unused edges | Self-loop connections ensure every node participates |

---

## 6. VETO Performance

The current model has demonstrated strong potential as a **VETO** system for CODEX-b, effectively discriminating between Standard Model background events and LLP signal events. The combination of:
- Physically motivated graph construction encoding VELO detector topology
- Interaction Network message passing for relational reasoning
- Jumping Knowledge pooling across all layers for multi-resolution features
- Robust I/O pipeline optimised for Lustre parallel filesystem

...results in a robust, scalable, and physically motivated veto system capable of real-time background rejection for the CODEX-b experiment.

---

*This design integrates particle physics principles (track linearity in VELO) with graph architectures (Interaction Networks) and a data pipeline optimised for parallel filesystems (Lustre), resulting in a robust, scalable, and physically motivated veto system for CODEX-b.*
