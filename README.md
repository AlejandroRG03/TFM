# Active Veto for CODEX-b via GNNs and LHCb VELO Hits

This repository contains the development of an Active Veto system based on Graph Neural Networks (GNN) for the CODEX-b experiment located in the LHCb cavern (IP8). The primary objective is to discriminate in real-time whether an event contains Standard Model particles (background) pointing toward the detector acceptance, allowing for their rejection and ensuring a "zero-background" environment for Long-Lived Particle (LLP) searches.

The current model has demonstrated strong VETO potential, achieving high signal efficiency with excellent background rejection on held-out test data for both muon and $K_L^0$ backgrounds.

## Project Motivation

CODEX-b searches for Beyond the Standard Model (BSM) signatures manifesting as displaced decays. To maximize sensitivity, it is crucial to identify background particles (such as muons and $K_L^0$) before they reach the detector volume. By utilizing the raw "hits" from the Vertex Locator (VELO) subdetector, we can reconstruct trajectory patterns and veto background events with high efficiency.

## Dataset and Features

The data consists of simulated VELO hits for both signal and background events. Each hit includes raw spatial coordinates (centered on the beamspot), module identifiers, and cluster properties. Engineered features include cylindrical coordinates ($r_T$, $\phi$), the projective `codex_angle` toward CODEX-b, and event-level multiplicity variables.

## Classifier Architecture (GNN)

The main implementation is located in the `/GNN` directory and utilizes PyTorch Geometric (PyG). The model is based on an **Interaction Network** designed to exploit the detector topology and hit correlations.

### GNN Components:

1. **Graph Construction:** Hits are transformed into nodes. Edges are created based on VELO detector geometry — intra-module radius graphs, inter-module KNN to adjacent planes (M±1, M±2), encoding physically plausible track segments.
2. **Interaction Network:** Message-passing layers where an Edge MLP computes messages from concatenated source/destination node features + edge attributes, and a Node MLP updates each node's state via residual connections.
3. **Jumping Knowledge Pooling:** Features are pooled from every Interaction layer using both AttentionalAggregation and GlobalMaxPool, providing multi-resolution graph representations.
4. **Classifier:** A multi-layer MLP with Dropout that produces a binary signal vs. background score.

### Key Features:
- **Static, precomputed graphs** — built once on CPU during data preparation, eliminating dynamic graph construction during training.
- **10-dimensional geometric edge attributes** — encode spatial differences, distances, cylindrical deltas, and unit direction vectors.
- **Self-loops** — each node connects to itself to retain its own state through message passing.
- **BF16 mixed precision** — prevents overflow in squared spatial differences.
- **CosineAnnealingWarmRestarts** scheduler with linear warmup.

## Directory Structure

```
GNN/
├── codex_gnn_model.py      # Interaction Network model definition
├── lightning_model.py       # PyTorch Lightning wrapper (optimizer, metrics, logging)
├── lightning_train.py       # Training pipeline (IterableDataset, DataLoader, Trainer)
├── gnn_tests.py             # Inference & evaluation on held-out test set
├── repack_chunks.py         # Converts legacy chunk format to repacked tensor-dict format
├── current_model.md         # Full technical report
├── utils/
│   ├── build_graph.py       # VELO graph construction (intra/inter/skip edges)
│   ├── prepare_torch_files.py  # ROOT → repacked PyG chunk pipeline
│   ├── obtain_stats.py      # Dask-based normalization statistics computation
│   └── ...
├── models/{KL0,MUON}/       # Trained checkpoints
└── test_plots/              # Evaluation plots
```

*This implementation is currently in active development.*
