import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


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

