"""
CODEX-b Veto Active GNN — InteractionNetwork Architecture (V4.1)

Key design decisions vs V3:
  - Module embedding removed (not useful for discrimination).
  - Continuous features reduced from 9 to 8 (eta dropped — r=0.77 with codex_angle).

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
from torch.nn import Linear, Sequential, SiLU, LayerNorm, Dropout, ModuleList
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
        n_cont_features: Number of continuous node features (default 8).
        hidden_channels: Width of the main representation (default 128).
        edge_dim:        Dimensionality of precomputed edge attributes (default 10).
        global_dim:      Number of global event-level features (default 3).
        num_layers:      Number of InteractionNetwork layers (default 5).
        edge_hidden:     Width of the edge MLP bottleneck (default 96).
    """
    def __init__(self, n_cont_features=8,
                 hidden_channels=96, edge_dim=10, global_dim=3, num_layers=4,
                 edge_hidden=64):
        super().__init__()
        self.num_layers = num_layers
        self.jk_mid_layer = num_layers // 2 - 1 # which layer to pool from for Jumping Knowledge (0-based index)

        # 1. Initial encoder: projects continuous features → hidden
        self.node_encoder = Sequential(
            Linear(n_cont_features, hidden_channels),
            LayerNorm(hidden_channels),
            SiLU()
        )

        # 2. Edge encoder: normalises raw edge features (mm-scale → stable scale)
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

        # 4. Global Attention Pooling
        # V4.1: we use a separate attention pool for mid layer and final layer outputs
        # to capture multi-scale features
        self.attn_pools = ModuleList()
        for _ in range(2): # 2, independen of num_layers
            gate_nn = Sequential(
                Linear(hidden_channels, hidden_channels // 2),
                SiLU(),
                Linear(hidden_channels // 2, 1)
            )
            self.attn_pools.append(aggr.AttentionalAggregation(gate_nn))

        # 5. Final Classifier (MLP)
        # Jumping Knowledge: we concatenate pooled features from ALL layers
        # (pool_att + pool_max) * num_layers + global_attr
        pool_dim = (hidden_channels * 2) * 2 + global_dim
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
        batch     = data.batch
        global_attr = data.global_attr
        edge_index  = data.edge_index
        edge_attr   = data.edge_attr

        # ── Initial Encoding ─────────────────────────────────────────────
        x = self.node_encoder(x_cont)
        edge_attr_enc = self.edge_encoder(edge_attr)

        # ── Global Pooling (Jumping Knowledge) ────────────────────────
        
        pooled_features = []
        
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr_enc)
            
            # JK extraction: Middle layer (local pseudo-tracks)
            if idx == self.jk_mid_layer:
                pooled_features.append(self.attn_pools[0](x, batch))
                pooled_features.append(global_max_pool(x, batch))
            
            # JK extraction: Final layer (global structure)
            elif idx == self.num_layers - 1:
                pooled_features.append(self.attn_pools[1](x, batch))
                pooled_features.append(global_max_pool(x, batch))


        
        # Combine all levels + global attributes (nVtx, nClu, etc.)
        out = torch.cat(pooled_features + [global_attr], dim=-1)
        return self.classifier(out)