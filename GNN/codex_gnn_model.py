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
    def __init__(self, node_dim, edge_dim, edge_hidden=96, dropout=0.3):
        super().__init__(aggr='add')

        # Edge MLP: computes message from source node, target node, and edge features
        self.edge_mlp = Sequential(
            Linear(2 * node_dim + edge_dim, edge_hidden),
            LayerNorm(edge_hidden),
            SiLU(),
            Linear(edge_hidden, node_dim),
            Dropout(dropout)
        )

        # Node MLP: updates node from old state + aggregated messages
        self.node_mlp = Sequential(
            Linear(2 * node_dim, node_dim),
            SiLU(),
            Linear(node_dim, node_dim),
            Dropout(dropout)
        )

        self.ln_node = LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # Pre-LN: apply LayerNorm BEFORE the message passing and update
        x_norm = self.ln_node(x)
        
        # Message passing: aggregate messages
        agg = self.propagate(edge_index, x=x_norm, edge_attr=edge_attr)
        
        # Node update with residual connection
        x_new = self.node_mlp(torch.cat([x_norm, agg], dim=-1))
        return x + x_new


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
        n_cont_features: Number of continuous node features (default 11).
        n_modules:       Number of distinct VELO modules for the embedding (default 52).
        embedding_dim:   Dimensionality of the module embedding (default 24).
        hidden_channels: Width of the main representation (default 128).
        edge_dim:        Dimensionality of precomputed edge attributes (default 10).
        global_dim:      Number of global event-level features (default 3).
        num_layers:      Number of InteractionNetwork layers (default 5).
        edge_hidden:     Width of the edge MLP bottleneck (default 96).
    """
    def __init__(self, n_cont_features=11, n_modules=52, embedding_dim=24,
                 hidden_channels=128, edge_dim=10, global_dim=3, num_layers=5,
                 edge_hidden=96):
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
        edge_enc_dim = 48  # compact learned edge representation
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

        # 4. Readout Mechanisms
        self.attn_pools = ModuleList([
            aggr.AttentionalAggregation(Sequential(
                Linear(hidden_channels, hidden_channels // 2),
                SiLU(),
                Linear(hidden_channels // 2, 1)
            )) for _ in range(2)
        ])

        # 5. Final Classifier (MLP)
        # Simplified Jumping Knowledge: we use Attentional + Max pooling
        # from a subset of layers (e.g. layers 3 and 5) to save memory/speed.
        # (pool_attn + pool_max) * 2 layers + global_attr
        pool_dim = (hidden_channels * 2) * 2 + global_dim
        self.classifier = Sequential(
            Linear(pool_dim, hidden_channels * 4), # Wider first layer
            SiLU(),
            Dropout(0.3),
            Linear(hidden_channels * 4, hidden_channels),
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
        selected_layers = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr_enc)
            if i in (2, self.num_layers - 1):
                selected_layers[i] = x

        # ── Global Pooling (Simplified Jumping Knowledge) ─────────────
        # We aggregate information from layers 3 and 5 (last layer).
        # This captures both intermediate track fragments and final representation.
        selected_layers = [selected_layers[2], selected_layers[self.num_layers - 1]]
        
        pooled_features = []
        for i, x_layer in enumerate(selected_layers):
            pooled_features.append(self.attn_pools[i](x_layer, batch))
            pooled_features.append(global_max_pool(x_layer, batch))
        
        # Combine selected levels + global attributes
        out = torch.cat(pooled_features + [global_attr], dim=-1)
        return self.classifier(out)