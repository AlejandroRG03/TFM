import torch
from torch_geometric.data.collate import collate
import time
import sys

def test_repack(filepath):
    print(f"Loading {filepath} ...")
    t0 = time.time()
    data_list = torch.load(filepath, weights_only=False, map_location='cpu')
    t1 = time.time()
    print(f"Loaded {len(data_list)} events in {t1-t0:.2f} seconds.")
    
    print("Collating...")
    t2 = time.time()
    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False
    )
    t3 = time.time()
    print(f"Collated in {t3-t2:.2f} seconds.")
    
    repacked_path = filepath.replace('.pt', '_repacked.pt')
    print(f"Saving to {repacked_path} ...")
    t4 = time.time()
    torch.save((data, slices), repacked_path)
    t5 = time.time()
    print(f"Saved in {t5-t4:.2f} seconds.")
    
    print("Testing load of repacked file...")
    t6 = time.time()
    # We can use mmap=True because it's just a dict of contiguous tensors now!
    data_loaded, slices_loaded = torch.load(repacked_path, map_location='cpu', weights_only=False, mmap=True)
    t7 = time.time()
    print(f"Loaded repacked file in {t7-t6:.4f} seconds.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_repack(sys.argv[1])
    else:
        import glob
        files = glob.glob("/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/graphs_*.pt")
        if files:
            test_repack(files[0])
