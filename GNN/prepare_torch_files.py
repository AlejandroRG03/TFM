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

    # the last event in the chunks may be incomplete, so we keep track of it to merge with the next chunk
    leftover_df = pd.DataFrame()

    # Iterate in chunks to avoid RAM saturation (100MB chunks)
    for chunk in uproot.iterate(FULL_PATH, VAR_NAMES, step_size="100 MB", library="pd"):

        # --- 0. HANDLE LEFTOVER FROM PREVIOUS CHUNK ---

        if not leftover_df.empty:
            chunk = pd.concat([leftover_df, chunk], ignore_index=True)
            
        # 2. Identify the last event in this chunk (may be incomplete)
        last_event_id = chunk['eventNumber'].iloc[-1]
        
        # 3. Split the chunk into complete events and leftover
        is_last_event = (chunk['eventNumber'] == last_event_id)
    
        leftover_df = chunk[is_last_event].copy()    # save it from next iteration
        df_to_process = chunk[~is_last_event].copy() # complete events

        # Si por casualidad un evento fuera tan grande que ocupara todo el chunk
        if df_to_process.empty:
            continue

        
        # --- 1. FEATURE ENGINEERING ---
        # Spherical/Collider variables (Mathematically redundant, but shortcuts for the GNN)
        df_to_process['r_T'], df_to_process['eta'], df_to_process['phi'] = collider_system(df_to_process, x='x', y='y', z='z')
        df_to_process['codex_angle'] = compute_codex_angles(df_to_process, x='x', y='y', z='z')

        # --- 2. NORMALIZATION ---
        cont_cols = ['x', 'y', 'z', 'r_T', 'phi', 'eta', 'n_pix', 'codex_angle'] # standardize only numerical features, not categorical
        df_to_process[cont_cols] = (df_to_process[cont_cols] - df_to_process[cont_cols].mean()) / (df_to_process[cont_cols].std() + 1e-8)
        
        # Normalize global event variables
        global_cols = ['nVtx_per_event', 'nClu_per_event', 'nTrk_per_event']
        df_to_process[global_cols] = (df_to_process[global_cols] - df_to_process[global_cols].mean()) / (df_to_process[global_cols].std() + 1e-8)

        events = df_to_process.groupby("eventNumber")
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
            # Using spatial coordinates (eta, phi, z) for proximity-based graph construction
            # these features will help the GNN to build connections that tend to link hits belonging
            # to the same particle trajectory, since in VELO there are no magnetic fields, so particles
            # travel in straight lines, which means that eta and phi are approximately constant along the trajectory
            # while z changes when crossing different layers of the detector.
            coords = df_event[['eta', 'phi', 'z']].values.copy()
            coords_tensor = torch.tensor(coords, dtype=torch.float)
            
            edge_index = knn_graph(coords_tensor, k=K_NEIGHBOURS, loop=False)

            # --- E. Graph Label ---
            y_tensor = torch.tensor([IS_SIGNAL], dtype=torch.float)

            # --- F. PyG Data Object Construction ---
            graph = Data(
                x_cont=x_cont, 
                x_cat=x_cat,
                edge_index=edge_index, # connections based on spatial proximity
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