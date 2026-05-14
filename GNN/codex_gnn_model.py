"""
CODEX-b Veto Active GNN — InteractionNetwork Architecture (V3)

Key design decisions vs V2:
  - Static graph: edge_index + edge_attr are precomputed on CPU during data
    preparation (module-aware topology: intra/inter/skip edges).
  - InteractionNetwork layers replace GATv2Conv:
      • edge_mlp  ingests [x_src, x_dst, edge_attr] → rich relational features
      • node_mlp  ingests [x_old, aggregated_msgs]   → updated node state
    This avoids O(E·H) attention tensors and gives edge attributes first-class
    treatment (they enter the MLP directly, not just as attention bias).
  - No metric_mlp / knn_graph in the forward pass → zero non-differentiable ops.
  - Edge MLP uses a narrow bottleneck (96D) since it is applied per-edge
    (~millions per batch); node MLP can be wider since it is per-node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Embedding, Sequential, SiLU, LayerNorm, Dropout, ModuleList
from torch_geometric.nn import MessagePassing, aggr, global_max_pool, global_mean_pool


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction Network Layer
# ═══════════════════════════════════════════════════════════════════════════════

class InteractionLayer(MessagePassing):
    """
    Pure Interaction Network layer (no attention mechanism).

    Message function:
        m_{ij} = edge_mlp( [x_i ‖ x_j ‖ edge_attr_{ij}] )

    Update function:
        x_i' = node_mlp( [x_i ‖ Σ_j m_{ij}] )  +  x_i   (residual)
    """
    def __init__(self, node_dim, edge_dim, edge_hidden=96):
        super().__init__(aggr='add')

        # Edge MLP: computes message from source node, target node, and edge features
        # Uses a narrow bottleneck because it is applied PER EDGE (~millions per batch)
        self.edge_mlp = Sequential(
            Linear(2 * node_dim + edge_dim, edge_hidden),
            LayerNorm(edge_hidden),
            SiLU(),
            Linear(edge_hidden, node_dim)
        )

        # Node MLP: updates node from old state + aggregated messages
        # Can be wider because it is applied PER NODE (much fewer than edges)
        self.node_mlp = Sequential(
            Linear(2 * node_dim, node_dim),
            SiLU(),
            Linear(node_dim, node_dim)
        )

        self.ln = LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # Message passing: aggregate messages from neighbours
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Node update with residual connection
        x_new = self.node_mlp(torch.cat([x, agg], dim=-1))
        return self.ln(x + x_new)

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(inp)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class CODEXVetoGNN(nn.Module):
    """
    Graph-level binary classifier for CODEX-b Active Veto.

    Architecture:
        Node Encoder → 5× InteractionLayer → Multi-head Global Pooling → MLP Classifier

    Args:
        n_cont_features: Number of continuous node features (default 9).
        n_modules:       Number of distinct VELO modules for the embedding (default 52).
        embedding_dim:   Dimensionality of the module embedding (default 16).
        hidden_channels: Width of the main representation (default 128).
        edge_dim:        Dimensionality of precomputed edge attributes (default 10).
        global_dim:      Number of global event-level features (default 3).
        num_layers:      Number of InteractionNetwork layers (default 5).
        edge_hidden:     Width of the edge MLP bottleneck (default 96).
    """
    def __init__(self, n_cont_features=9, n_modules=52, embedding_dim=16,
                 hidden_channels=96, edge_dim=10, global_dim=3, num_layers=4,
                 edge_hidden=64):
        super().__init__()
        self.num_layers = num_layers

        # 1. Embedding for the module ID
        self.module_emb = Embedding(n_modules + 1, embedding_dim)

        # 2. Initial encoder: projects [continuous ‖ module_emb] → hidden
        self.node_encoder = Sequential(
            Linear(n_cont_features + embedding_dim, hidden_channels),
            LayerNorm(hidden_channels),
            SiLU()
        )

        # 2.5 Edge encoder: normalises raw edge features (mm-scale → stable scale)
        # Prevents fp16 overflow from raw spatial differences
        edge_enc_dim = 32  # compact learned edge representation
        self.edge_encoder = Sequential(
            Linear(edge_dim, edge_enc_dim),
            LayerNorm(edge_enc_dim),
            SiLU()
        )

        # 3. Deep Interaction Stack (edge_mlp + node_mlp, no attention)
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                InteractionLayer(
                    node_dim=hidden_channels,
                    edge_dim=edge_enc_dim,
                    edge_hidden=edge_hidden
                )
            )

        # 4. Global Attention Pooling (Readout)
        gate_nn = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            SiLU(),
            Linear(hidden_channels // 2, 1)
        )
        self.global_pool = aggr.AttentionalAggregation(gate_nn)

        # 5. Final Classifier (MLP)
        # Jumping Knowledge: we concatenate pooled features from ALL layers
        # (pool_att + pool_max + pool_mean) * num_layers + global_attr
        pool_dim = (hidden_channels * 3) * num_layers + global_dim
        self.classifier = Sequential(
            Linear(pool_dim, hidden_channels * 2),
            SiLU(),
            Dropout(0.3),
            Linear(hidden_channels * 2, hidden_channels),
            SiLU(),
            Dropout(0.3),
            Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x_cont    = data.x_cont
        x_cat     = data.x_cat
        batch     = data.batch
        global_attr = data.global_attr
        edge_index  = data.edge_index
        edge_attr   = data.edge_attr

        # ── Node Encoding ─────────────────────────────────────────────
        emb = self.module_emb(x_cat)
        x = self.node_encoder(torch.cat([x_cont, emb], dim=-1))

        # ── Edge Encoding (normalise raw features for fp16 stability) ─
        edge_attr_enc = self.edge_encoder(edge_attr)

        # ── Deep Interaction Stack ────────────────────────────────────
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr_enc)
            layer_outputs.append(x)

        # ── Global Pooling (Jumping Knowledge) ────────────────────────
        # We aggregate information from every depth level to capture 
        # both local track fragments and global event shape.
        pooled_features = []
        for x_layer in layer_outputs:
            pooled_features.append(self.global_pool(x_layer, batch))
            pooled_features.append(global_max_pool(x_layer, batch))
            pooled_features.append(global_mean_pool(x_layer, batch))
        
        # Combine all levels + global attributes (nVtx, nClu, etc.)
        out = torch.cat(pooled_features + [global_attr], dim=-1)
        return self.classifier(out)