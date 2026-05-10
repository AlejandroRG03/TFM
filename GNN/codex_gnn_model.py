import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, Sequential, SiLU, LayerNorm, Dropout, ModuleList
from torch_geometric.nn import GATv2Conv, aggr, global_max_pool, global_mean_pool, knn_graph
import os
import glob
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

class CODEXVetoGNN(torch.nn.Module):
    def __init__(self, n_cont_features=9, n_modules=52, embedding_dim=16, hidden_channels=128, global_dim=3, k=8, num_layers=5):
        """
        Deep Track-Aware GNN (V2)
        - Increased depth (5 layers)
        - Increased capacity (128 channels, 4 heads)
        - Robust residual stacks with LayerNorm
        """
        super(CODEXVetoGNN, self).__init__()
        self.k = k
        self.num_layers = num_layers
        
        # 1. Embedding for the module ID
        self.module_emb = Embedding(n_modules + 1, embedding_dim)
        
        # 2. Initial encoder for hit features
        self.node_encoder = Sequential(
            Linear(n_cont_features + embedding_dim, hidden_channels),
            LayerNorm(hidden_channels),
            SiLU()
        )
        
        # 2.5 Metric Learning MLP (Higher capacity for better topology)
        self.metric_mlp = Sequential(
            Linear(n_cont_features + embedding_dim, hidden_channels // 2),
            SiLU(),
            Linear(hidden_channels // 2, 8) # 8D latent space
        )
        
        # 3. Deep Interaction Stack
        self.convs = ModuleList()
        self.lns = ModuleList()
        
        head_dim = hidden_channels // 4 # Using 4 heads
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_channels, head_dim, heads=4, concat=True, edge_dim=16)
            )
            self.lns.append(LayerNorm(hidden_channels))
        
        # 4. Global Attention Pooling (Readout)
        gate_nn = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            SiLU(),
            Linear(hidden_channels // 2, 1)
        )
        self.global_pool = aggr.AttentionalAggregation(gate_nn)
        
        # 5. Final Classifier (MLP)
        self.classifier = Sequential(
            Linear(hidden_channels * 3 + global_dim, hidden_channels),
            SiLU(),
            Dropout(0.3),
            Linear(hidden_channels, hidden_channels // 2),
            SiLU(),
            Dropout(0.3),
            Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x_cont, x_cat, batch, global_attr = \
            data.x_cont, data.x_cat, data.batch, data.global_attr
        
        # Initial projection
        emb = self.module_emb(x_cat)
        x_concat = torch.cat([x_cont, emb], dim=-1)
        
        x = self.node_encoder(x_concat)
        
        # --- METRIC LEARNING GRAPH CONSTRUCTION ---
        latent_coords = self.metric_mlp(x_concat)
        
        # Use a hybrid approach: Physical (normalized) + Latent space for graph construction.
        # x_cont[:, :3] contains the normalized x, y, z coordinates (mean 0, std 1).
        # This matches the scale of the latent embeddings, ensuring the graph is not biased by units (mm vs latent).
        pos_normalized = x_cont[:, :3]
        combined_coords = torch.cat([pos_normalized, latent_coords], dim=-1)
        
        # Build k-NN graph in the combined space (11D)
        edge_index = knn_graph(combined_coords, k=self.k, batch=batch, loop=False)
        
        # 1. Physical edge attributes (geometry) - Using normalized coordinates for stability
        coords_diff = x_cont[edge_index[0], :3] - x_cont[edge_index[1], :3]
        dist = torch.norm(coords_diff, p=2, dim=-1, keepdim=True) + 1e-6
        dir_norm = coords_diff / dist
        
        latent_diff = latent_coords[edge_index[0]] - latent_coords[edge_index[1]]
        latent_dist = torch.norm(latent_diff, p=2, dim=-1, keepdim=True)

        edge_attr = torch.cat([
            coords_diff, dir_norm, dist,
            latent_diff, latent_dist
        ], dim=-1)

        # Deep Interaction Stack with Residual Connections
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.lns[i](x)
            x = F.silu(x + x_res)

        # Global Pooling
        pool_att = self.global_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_mean = global_mean_pool(x, batch)
        
        # Classifier
        out = torch.cat([pool_att, pool_max, pool_mean, global_attr], dim=-1)
        return self.classifier(out)