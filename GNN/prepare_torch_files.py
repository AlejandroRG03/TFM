import uproot
import numpy as np
import pandas as pd
import os
import torch

# Set the input file name (without path)

input_file_name = "ntuple_signal_40114060.root"
is_signal = 1  # Set to 1 for signal, 0 for background

###############################################################################

input_file_path = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
output_dir = "/lustre/LHCb/alejandro.rodriguez/torch_data"
output_name = "processedPytTorch_" + input_file_name.replace(".root", ".pt")
tree_name = "VeloMultiTuple_73eaa531/Clusters"


var_names = ['eventNumber', 'x', 'y', 'z', 'n_pix', 'module', 'nVtx_per_event', 'nClu_per_event']

processed_events = 0

full_path = f"{input_file_path}{input_file_name}:{tree_name}"


print(f"Reading data from {input_file_path}{input_file_name}...")

for chunk in uproot.iterate(full_path, var_names, step_size="100MB", library="pd"):

    events = chunk.groupby("eventNumber")

    for event_id, df_event in events:

        # Create a tensor for the current event, node features are x, y, z, n_pix, module
        x_tensor = torch.tensor(df_event[['x', 'y', 'z', 'n_pix', 'module']].values, dtype=torch.float)
        y_tensor = torch.tensor([is_signal], dtype=torch.long)

        data_dict = {
            'x': x_tensor,
            'y': y_tensor,
            'eventNumber': event_id
        }
        
        torch.save(data_dict, os.path.join(output_dir, f"event_{event_id}.pt"))

print("Data processing complete. Tensors saved to:", output_dir + "/" + output_name)