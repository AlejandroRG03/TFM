from functions import *
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
INPUT_FILE_NAME = "ntuple_background_38011800.root"
IS_SIGNAL = 0
DEC_ID    = "38011800"

INPUT_FILE_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
OUTPUT_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"

# Variables to extract from ROOT
VAR_NAMES = [
    'eventNumber', 'x', 'y', 'z', 'n_pix', 'module', 
    'nVtx_per_event', 'nClu_per_event', 'nTrk_per_event'
]
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"
FULL_PATH = f"{INPUT_FILE_PATH}{INPUT_FILE_NAME}:{TREE_NAME}"

K_NEIGHBOURS = 5  

# ==============================================================================
# PROCESSING PIPELINE
# ==============================================================================
def prepare_torch_files():

    data_type = "signal" if IS_SIGNAL else "background"
    specific_output_dir = os.path.join(OUTPUT_DIR, data_type, DEC_ID)
    os.makedirs(specific_output_dir, exist_ok=True)
    print(f"Starting processing from: {FULL_PATH}")

    chunk_counter = 0
    total_events = 0

    # Iterate in chunks to avoid RAM saturation (100MB chunks)
    for chunk in uproot.iterate(FULL_PATH, VAR_NAMES, step_size="100 MB", library="pd"):
        
        # --- 1. FEATURE ENGINEERING ---
        # Spherical/Collider variables (Mathematically redundant, but shortcuts for the GNN)
        chunk['r_T'], chunk['eta'], chunk['phi'] = collider_system(chunk, x='x', y='y', z='z')
        chunk['codex_angle'] = compute_codex_angles(chunk, x='x', y='y', z='z')

        # --- 2. NORMALIZATION ---
        cont_cols = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle'] # standardize only numerical features, not categorical
        chunk[cont_cols] = (chunk[cont_cols] - chunk[cont_cols].mean()) / (chunk[cont_cols].std() + 1e-8)
        
        # Normalize global event variables
        global_cols = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']
        chunk[global_cols] = (chunk[global_cols] - chunk[global_cols].mean()) / (chunk[global_cols].std() + 1e-8)

        events = chunk.groupby("eventNumber")
        chunk_data_list = []

        for event_id, df_event in events:
            # Quality filter: minimum hits to form a graph with K neighbors
            if len(df_event) <= K_NEIGHBOURS:
                continue

            # --- A. Normalized Continuous Variables (Float) ---
            x_cont = torch.tensor(df_event[cont_cols].values, dtype=torch.float)
            
            # --- B. Categorical Variable (Long) for Embeddings ---
            # module is NOT normalized, passed raw as an ID
            x_cat = torch.tensor(df_event['module'].values, dtype=torch.long)
            
            # --- C. Global Variables (Float) ---
            # Take the first row as it's the same value for the whole event
            global_attr = torch.tensor(df_event[global_cols].iloc[0].values, dtype=torch.float).unsqueeze(0)

            # --- D. Edge Creation (k-NN) ---
            # Use raw coordinates ONLY to calculate distances, 
            # but divide Z by 10 so neighbors are searched more spherically and
            # follow track shapes, mitigating VELO elongation.
            coords = df_event[['x', 'y', 'z']].values.copy()
            coords[:, 2] = coords[:, 2] / 10.0  # Scale Z
            coords_tensor = torch.tensor(coords, dtype=torch.float)
            
            edge_index = knn_graph(coords_tensor, k=K_NEIGHBOURS, loop=False)

            # --- E. Graph Label ---
            y_tensor = torch.tensor([IS_SIGNAL], dtype=torch.float)

            # --- F. PyG Data Object Construction ---
            graph = Data(
                x_cont=x_cont, 
                x_cat=x_cat,
                edge_index=edge_index, 
                y=y_tensor,
                global_attr=global_attr,
                event_id=torch.tensor([event_id], dtype=torch.long)
            )
            
            chunk_data_list.append(graph)
            total_events += 1
            
        # Save to Lustre
        chunk_filename = os.path.join(specific_output_dir, f"graphs_{chunk_counter}.pt")
        torch.save(chunk_data_list, chunk_filename)
        print(f"-> Chunk {chunk_counter} saved with {len(chunk_data_list)} events (Total: {total_events})")
        
        chunk_counter += 1

    print(f"\nProcessing completed! {total_events} graphs ready in {specific_output_dir}")

if __name__ == "__main__":
    prepare_torch_files()