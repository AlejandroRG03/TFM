# Active Veto for CODEX-b via GNNs and LHCb VELO Hits

This repository contains the development of an Active Veto system based on Graph Neural Networks (GNN) for the CODEX-beta and CODEX-b experiments located in the LHCb cavern (IP8). The primary objective is to discriminate in real-time whether an event contains Standard Model particles (background) pointing toward the detector acceptance, allowing for their rejection and ensuring a "zero-background" environment for Long-Lived Particle (LLP) searches.

## Project Motivation

CODEX-b searches for Beyond the Standard Model (BSM) signatures manifesting as displaced decays. To maximize sensitivity, it is crucial to identify background particles (such as muons and $K_L^0$) before they reach the detector volume. By utilizing the raw "hits" from the Vertex Locator (VELO) subdetector, we can reconstruct trajectory patterns and veto background events with high efficiency.

## Dataset and Features

The data consists of simulated VELO hits for both signal and background events. Each hit includes the following variables (features):

* **Temporal:** `bxType`, `bxId`, `gpsTime`.
* **Identifiers:** `eventNumber`, `runNumber`, `triggerType`, `eventType`.
* **Global Geometry:** `beamspotX`, `beamspotY`.
* **Multiplicity:** `nTrk_per_event`, `nVtx_per_event`, `nClu_per_event`.
* **Hardware Information:** `module`, `chip`, `sensor`, `row`, `col`.
* **Cluster Properties:** `n_pix` (correlated with the angle of incidence).
* **Spatial Coordinates (3D):** `x`, `y`, `z` (Cartesian coordinates in the LHCb reference frame).

## Classifier Architecture (GNN)

The main implementation is located in the `/GNN` directory and utilizes PyTorch Geometric (PyG). The model is based on an Interaction Network (IN) designed to exploit the detector topology and hit correlations.

### GNN Components:

1. **Graph Construction:** Hits are transformed into nodes. Edges are created based on geometric constraints, such as proximity between VELO planes and alignment with the CODEX-b axis.
2. **Edge Network (Relational Model):** An MLP that predicts the importance of connections (edges) between hits, effectively performing edge classification or weighting.
3. **Node Network (Object Model):** Updates the latent state of each hit based on aggregated messages received from its neighbors.
4. **Global Pooling:** Aggregation of node and edge information from the entire graph to produce a binary classification score (Signal vs. Background).

*Notice that this implementation is currently in development and is subject to changes*
