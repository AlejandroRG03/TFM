import os
import glob
import torch
import random
import torch.nn as nn
from torch.optim import Adam # adam is a powerful optimizer
from torch_geometric.loader import DataLoader
from codex_gnn_model import CODEXVetoGNN
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

print(f"--> CUDA? {torch.cuda.is_available()}")
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================

DATA_DIR = "/lustre/LHCb/alejandro.rodriguez/torch_data"

# for dec ids, use list in case we change the approach to multi-class classification, for now only one type of signal and one type of background
# WARNING: if we want to do multi-class classification, we need to change the model output layer and the loss function accordingly!
SIGNAL_DEC_IDS      = ["40114060"] 
BACKGROUND_DEC_IDS  = ["38011800"]

OUTPUT_NAME = f"S{SIGNAL_DEC_IDS[0]}_B{BACKGROUND_DEC_IDS[0]}"

BATCH_SIZE    = 256
EPOCHS        = 20
LEARNING_RATE = 1e-3
MAX_CHUNKS    = 2      # Set to None to use all data, WARNING: only if bkg and sig have the same number of chunks!
TRAIN_SPLIT   = 0.8
PATIENCE      = 10

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_files(data_dir, dec_ids, data_type):
    files = []
    for dec_id in dec_ids:
        path = os.path.join(data_dir, data_type, dec_id, "*.pt")
        files.extend(glob.glob(path))
    return files

def get_paired_files(sig_list, bkg_list):
    """
    Pairs signal and background files for training. If one list is shorter, it will cycle through it.
    """
    if not sig_list or not bkg_list:
        raise ValueError("[ERROR] No signal or background files found.")

    if len(sig_list) > len(bkg_list):
        return list(zip(sig_list, itertools.cycle(bkg_list)))
    else:
        return list(zip(itertools.cycle(sig_list), bkg_list))

def run_epoch(model, loader_files, device, criterion, optimizer=None):
    """Runs a single epoch of training or validation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_graphs = 0, 0
    y_true, y_prob = [], []

    # loader_files is a list of tuples (sig_file, bkg_file)
    for sig_file, bkg_file in loader_files:
        combined_data = torch.load(sig_file, weights_only=False) + torch.load(bkg_file, weights_only=False)
        loader = DataLoader(combined_data, batch_size=BATCH_SIZE, shuffle=is_train, num_workers=4)

        for batch in loader:
            batch = batch.to(device)
            if is_train: optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                out = model(batch).squeeze(-1)
                loss = criterion(out, batch.y)
                
                if is_train:
                    loss.backward()   # backpropagation
                    optimizer.step()  # update weights

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            
            y_prob.extend(torch.sigmoid(out).detach().cpu().numpy())
            y_true.extend(batch.y.detach().cpu().numpy())

    if total_graphs == 0: raise ValueError("[ERROR] No graphs in batch.")
    avg_loss = total_loss / total_graphs
    acc = accuracy_score(y_true, (np.array(y_prob) > 0.5).astype(int))
    try: auc = roc_auc_score(y_true, y_prob)
    except ValueError: auc = 0.5

    return avg_loss, acc, auc

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Device: {device}")

    # 1. Data Preparation
    sig_files = get_files(DATA_DIR, SIGNAL_DEC_IDS, "signal")
    bkg_files = get_files(DATA_DIR, BACKGROUND_DEC_IDS, "background")

    if not sig_files or not bkg_files:
        print("[ERROR] No data files found.")
        return

    if MAX_CHUNKS is not None:
        sig_files, bkg_files = sig_files[:MAX_CHUNKS], bkg_files[:MAX_CHUNKS]

    random.shuffle(sig_files)
    random.shuffle(bkg_files)

    n_train_sig = max(1, int(TRAIN_SPLIT * len(sig_files)))
    n_train_bkg = max(1, int(TRAIN_SPLIT * len(bkg_files)))

    sig_train = sig_files[:n_train_sig]
    bkg_train = bkg_files[:n_train_bkg]

    val_files   = get_paired_files(sig_files[n_train_sig:], bkg_files[n_train_bkg:])

    print(f"--> Dataset: {len(sig_train)} train chunks, {len(val_files)} val chunks")
    

    # 2. Model, Loss, Optimizer
    model = CODEXVetoGNN().to(device)
    pos_weight = torch.tensor([n_train_sig / n_train_bkg]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        random.shuffle(sig_train)
        random.shuffle(bkg_train)

        train_files = get_paired_files(sig_train, bkg_train)
        # Train
        t_loss, t_acc, t_auc = run_epoch(model, train_files, device, criterion, optimizer)
        
        # Validation
        v_loss, v_acc, v_auc = run_epoch(model, val_files, device, criterion)

        # Logging
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | AUC: {t_auc:.4f}")
        print(f"  Val   | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | AUC: {v_auc:.4f}")

        # Checkpointing
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{OUTPUT_NAME}_best.pth")
            print(f"  --> Best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\n[Early Stopping] No improvement for {PATIENCE} epochs.")
            break

if __name__ == "__main__":
    train()