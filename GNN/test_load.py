import torch
import time
import sys

def test_load(filepath):
    print(f"Loading {filepath} ...")
    t0 = time.time()
    try:
        data = torch.load(filepath, weights_only=False, map_location='cpu')
        t1 = time.time()
        print(f"Loaded {len(data)} events in {t1-t0:.2f} seconds.")
        print(f"First event: {data[0]}")
    except Exception as e:
        print(f"Failed to load: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_load(sys.argv[1])
    else:
        import glob
        files = glob.glob("/lustre/LHCb/alejandro.rodriguez/torch_data/signal/40114060/*.pt")
        if files:
            test_load(files[0])
