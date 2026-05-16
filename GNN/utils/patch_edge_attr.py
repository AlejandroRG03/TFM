import os
import glob
import time
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"
N_WORKERS = 8


def patch_file(filepath):
    """Adds edge_attr to all Data objects in a chunk file if missing."""
    import torch
    from build_graph import compute_edge_attr

    try:
        data_list = torch.load(filepath, weights_only=False, map_location='cpu')

        if hasattr(data_list[0], 'edge_attr') and data_list[0].edge_attr is not None:
            return (filepath, False)

        for data in data_list:
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                data.edge_attr = compute_edge_attr(
                    data.pos, data.x_cont, data.edge_index.to(torch.long)
                )

        torch.save(data_list, filepath)
        return (filepath, True)

    except Exception as e:
        print(f"Error patching {os.path.basename(filepath)}: {e}", flush=True)
        return (filepath, None)


def main():
    files = glob.glob(os.path.join(DATA_DIR, "**/*.pt"), recursive=True)
    print(f"Found {len(files)} files. Processing with {N_WORKERS} workers...")

    t0 = time.time()
    patched = 0
    skipped = 0
    errors = 0

    ctx = get_context('fork')
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(patch_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Patching"):
            _, ok = future.result()
            if ok:
                patched += 1
            elif ok is None:
                errors += 1
            else:
                skipped += 1

    t = time.time() - t0
    print(f"\nDone in {t/60:.1f} min | "
          f"Patched: {patched} | Skipped: {skipped} | Errors: {errors}")


if __name__ == "__main__":
    main()
