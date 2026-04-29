import uproot
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
input_file_name = "ntuple_background_38011800.root" 
is_signal = 0  # 1 signal, 0 background
dec_id    = "38011800" # new folder for this dec_id, avoid mixing data

input_file_path = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
output_dir = "/lustre/LHCb/alejandro.rodriguez/torch_data"

# Variables
var_names = ['eventNumber', 'x', 'y', 'z', 'n_pix', 'module', 'nVtx_per_event', 'nClu_per_event']
tree_name = "VeloMultiTuple_73eaa531/Clusters"
full_path = f"{input_file_path}{input_file_name}:{tree_name}"

# Graph hyperparameters
K_NEIGHBOURS = 5  # Maximum number of edges per node (k-NN)

# ==============================================================================
# PROCESSING PIPELINE
# ==============================================================================
def prepare_torch_files():
    data_type = "signal" if is_signal else "background"
    specific_output_dir = os.path.join(output_dir, data_type, dec_id)
    os.makedirs(specific_output_dir, exist_ok=True)
    print(f"Starting processing from: {full_path}")

    chunk_counter = 0
    total_events = 0

    # 1. Read in chunks to limit RAM usage
    for chunk in uproot.iterate(full_path, var_names, step_size="100 MB", library="pd"):
        
        events = chunk.groupby("eventNumber")
        chunk_data_list = [] # List to store graphs for this chunk

        for event_id, df_event in events:
            
            # Quality filter: we need at least (K_NEIGHBOURS + 1) hits to 
            # form a meaningful graph and avoid KNN errors (without self-connections).
            if len(df_event) <= K_NEIGHBOURS: # number of nodes > K_NEIGHBOURS
                continue

            # 2. Node feature extraction (All features)
            # Normalizing these values later will be key, but for now we extract them raw
            node_features = torch.tensor(df_event[['x', 'y', 'z', 'n_pix', 'module']].values, dtype=torch.float)
            
            # 3. Edge construction
            # Extract ONLY physical coordinates to calculate real distances
            coords = torch.tensor(df_event[['x', 'y', 'z']].values, dtype=torch.float)
            
            # k-NN creates connections (edge_index). loop=False prevents a hit from connecting to itself
            edge_index = knn_graph(coords, k=K_NEIGHBOURS, loop=False) # connections are physical distances

            # Whole graph label
            y_tensor = torch.tensor([is_signal], dtype=torch.long) # class label for the entire graph (event)

            # 4. Creation of native PyG Graph object
            # Optionally you can save global variables that affect the entire event
            # globals = torch.tensor(df_event[['nVtx_per_event', 'nClu_per_event']].iloc[0].values, dtype=torch.float)
            
            graph = Data(
                x=node_features, 
                edge_index=edge_index, 
                y=y_tensor,
                event_id=torch.tensor([event_id], dtype=torch.long) # Useful for later debugging
                # globals=globals # Uncomment if you use global features in your GNN
            )
            
            chunk_data_list.append(graph)
            total_events += 1
            
        # 5. Save to lustre
        # Save the complete list of graphs in a single binary file
        chunk_filename = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
        
        torch.save(chunk_data_list, chunk_filename)
        print(f"-> Chunk {chunk_counter} saved with {len(chunk_data_list)} events (Total: {total_events})")
        
        chunk_counter += 1

    print(f"\nSuccess! Preprocessing completed. {total_events} graphs ready for PyTorch Geometric in {output_dir}")

if __name__ == "__main__":
    prepare_torch_files()