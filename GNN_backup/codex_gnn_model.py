import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
import os
import glob
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


"""
Problem statement:

Based on the information of the hits in the VELO subdetector, we want to classify 
the events that contain signal pointing towards CODEX-b, and the events that contain
background pointing towards CODEX-b.

We will use a Graph Neural Network (GNN) for this task, since they are capable of 
understanding tracking information and spatial relationships between hits.

In particular, we use knn_graph from pyTorch Geometric to construct the graph from the hit information,
and then we will use a simple GNN architecture (e.g. Graph Convolutional Network)
 to perform the classification.
"""


class CODEXVetoGNN(torch.nn.Module):
    def __init__(self, n_cont_features=8, n_modules=52, embedding_dim=16, hidden_channels=64, global_dim=3):

        """
        :param n_cont_features: Number of continuous features per hit (e.g. x, y, z, r_T, phi, eta, n_pix, codex_angle)
        :param n_modules: Number of unique module IDs (categorical feature for embedding), VELO has 52 modules
        :param embedding_dim: Dimension of the module ID embedding, features that the model will learn to represent the module information
        :param hidden_channels: Number of hidden units in the GNN layers, controls the capacity of the model to learn complex relationships. \
        Each hit has associated a feature vector of size n_cont_features + embedding_dim after concatenating continuous features and module embedding. \
        Using the node_encoder, we project this concatenated feature vector into a hidden space of dimension hidden_channels, which is the input dimension for the GNN layers.

        :param global_dim: Dimension of the global event-level attributes (e.g. nVtx, nClu, nTrk)
        """

        super(CODEXVetoGNN, self).__init__()
        
        # 1. Embedding for the module ID (Categorical feature)
        self.module_emb = Embedding(n_modules + 1, embedding_dim)
        # Explanation of Embeddings: module id is a categorical feature that indicates which module in the VELO detector the hit belongs to.
        # using embeddings, we allow the model to transform this categorical information into a feature vector of size embedding_dim, wich
        # enables the model to learn characteristics of the modules. For instance, this feature vector can be something like
        # [is_inner_module, is_outer_module, module_position_along_z, ...], which can help the model to understand the spatial distribution of hits.

        
        # 2. Initial encoder for hit features
        # x_cont: x, y, z, r_T, phi, eta, n_pix, codex_angle
        self.node_encoder = Sequential(
            Linear(n_cont_features + embedding_dim, hidden_channels), # project concatenated features to hidden dimension, 24 -> 64, linear means y = ax + b
            BatchNorm1d(hidden_channels),                             # normalize the features
            ReLU()                                                    # Activation function 
        )
        
        # 3. Graph Convolution layers (Message Passing) with 4 Layers and Residual Connections
        self.conv1 = SAGEConv(hidden_channels, hidden_channels) # SAGEConv is much faster and memory-efficient than GAT
        self.bn1   = BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels) 
        self.bn2   = BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3   = BatchNorm1d(hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn4   = BatchNorm1d(hidden_channels)
        
        # 4. Final Classifier (MLP) after pooling
        # Concatenate mean pooling, max pooling, and global event attributes
        self.classifier = Sequential(                                   # Sequential MLP, layer of linear transformations, activations
            Linear(hidden_channels * 2 + global_dim, hidden_channels),  # input layer, dimension is the concatenation of mean pooling, max pooling, and global attributes, 131 -> 64
            ReLU(),
            Dropout(0.3),  # Dropout for regularization
            Linear(hidden_channels, hidden_channels // 2),              # hidden layer, 64 -> 32
            ReLU(),
            Dropout(0.3),
            Linear(hidden_channels // 2, 1) # Logit output for Binary Cross Entropy (one number because it's binary classification), 32 -> 1
        )
        # mean pooling and max pooling since we do not care about the hits themselves, but about the overall event, 
        # so we need to aggregate the hit information into a single vector representation for the whole graph (event).

        # We use only one hidden layer in the classifier because the relations between the different nodes are already captured in the GNN layers (SAGEConv)

    def forward(self, data):
        """
        forward method defines how the input data flows through the model to produce the output.
        data is a batch of graphs (events) where each graph has node features (x_cont and x_cat), edge_index (graph structure), batch (which graph each node belongs to), 
        and global_attr (global event-level attributes).
        """

        x_cont, x_cat, edge_index, batch, global_attr = \
            data.x_cont, data.x_cat, data.edge_index, data.batch, data.global_attr
        
        # Project module embedding and concatenate with continuous features
        emb = self.module_emb(x_cat)
        x = torch.cat([x_cont, emb], dim=-1)
        
        x = self.node_encoder(x) # initial encoding of node features, 24 -> 64
        
        # Message Passing layers with residual connections (skip connections)
        x_res = x # residual connection, this helps the gradient flow through the network and prevents vanishing gradients
        x = self.bn1(self.conv1(x, edge_index)).relu() 
        x = x + x_res
        
        x_res = x
        x = self.bn2(self.conv2(x, edge_index)).relu()
        x = x + x_res
        
        x_res = x
        x = self.bn3(self.conv3(x, edge_index)).relu()
        x = x + x_res
        
        x_res = x
        x = self.bn4(self.conv4(x, edge_index)).relu()
        x = x + x_res
        
        # Global Pooling (Readout)
        # Obtain a vector representation for each graph (event) in the batch
        pool_mean = global_mean_pool(x, batch) # average of node features for each graph, captures overall trend of features across the graph
        pool_max = global_max_pool(x, batch) # max of node features for each graph, captures the most salient features across the graph
        
        # Concatenate global context (e.g., nVtx, nClu, nTrk)
        out = torch.cat([pool_mean, pool_max, global_attr], dim=-1) # dim=-1 means concatenate along the last dimension
        
        return self.classifier(out) # classify given the aggregated graph representation, output is a logit for binary classification