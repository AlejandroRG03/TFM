import os
import glob
import torch
from typing import Optional, Tuple, List
from torch_geometric.loader import DataLoader

from lightning_model import CODEXLightning
from lightning_train import ChunkIterableDataset, get_files, get_paired_files

from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import sys
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import set_tfm_style

set_tfm_style()

DATA_DIR = "/scratch/alejandro.rodriguez/torch_pileup"
SIGNAL_DEC_IDS = ["40114060"]
BKG_DEC_IDS = {"pileup_MUON": ["38000800"]}
MODEL_DIR = "/home3/alejandro.rodriguez/TFM/GNN/models"
N_TEST = 5

PLOTS_DIR = "test_plots"
PLOTS_DATA_DIR = "test_data"


def save_plot_data(bkg_type: str, plot_name: str, **data) -> None:
    os.makedirs(PLOTS_DATA_DIR, exist_ok=True)
    np.savez(os.path.join(PLOTS_DATA_DIR, f"{bkg_type}_{plot_name}.npz"), **data)


def load_test_data(bkg_type: str, batch_size: int = 256,
                   num_workers: int = 2) -> DataLoader:
    sig_files = get_files(DATA_DIR, SIGNAL_DEC_IDS, "signal")
    bkg_files = get_files(DATA_DIR, BKG_DEC_IDS[bkg_type], "background")
    relabel = False
    label_str = "Signal"

    sig_test = sig_files[-N_TEST:]
    bkg_test = bkg_files[-N_TEST:]
    test_pairs = get_paired_files(sig_test, bkg_test)

    print(f"{label_str} chunks: {len(sig_test)}  |  "
          f"Background chunks: {len(bkg_test)}  |  "
          f"Test pairs: {len(test_pairs)}")

    dataset = ChunkIterableDataset(test_pairs, is_validation=True,
                                   relabel_signal=relabel)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False,
        pin_memory=True,
    )
    return loader


def load_model(bkg_type: str,
               checkpoint_path: Optional[str] = None,
               device: str = "cuda") -> CODEXLightning:
    if checkpoint_path is None:
        model_dir = os.path.join(MODEL_DIR, bkg_type)
        pattern = os.path.join(model_dir, f"{bkg_type}_CODEX_GNN_best*.ckpt")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found: {pattern}")
        checkpoint_path = candidates[-1]
        mtime = os.path.getmtime(checkpoint_path)
        from datetime import datetime
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        size_mb = os.path.getsize(checkpoint_path) / 1e6
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"  Saved: {date_str}  |  Size: {size_mb:.0f} MB")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
    model = CODEXLightning.load_from_checkpoint(
        checkpoint_path, pos_weight_val=1.0, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: CODEXLightning,
                  loader: DataLoader
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Inference", unit="batch"):
        batch = batch.to(device)
        logits = model(batch)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch.y.cpu())
    return torch.cat(all_probs), torch.cat(all_labels).long()

def plot_probability_distributions(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str) -> None:
    save_plot_data(bkg_type, "probability_distributions",
                   probs=probs.numpy(), labels=labels.numpy())
    plt.figure(figsize=(8, 5))
    plt.hist(probs[labels == 1].numpy(), bins=50, alpha=0.5, label="Signal", color="blue", density=True)
    plt.hist(probs[labels == 0].numpy(), bins=50, alpha=0.5, label="Background", color="red", density=True)
    plt.title(f"Predicted Probabilities for {bkg_type} Background")
    plt.xlabel("Probability of Signal")
    plt.ylabel("PDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_probability_distributions.pdf")


def compute_accuracy(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).long()
    accuracy = (preds == labels).float().mean().item()
    print(f"Accuracy (threshold={threshold}): {accuracy:.4f}")
    return accuracy

def compute_roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    fpr, tpr, _ = metrics.roc_curve(labels.numpy(), probs.numpy())
    auc = metrics.auc(fpr, tpr)
    print(f"ROC AUC: {auc:.4f}")
    return auc

def compute_precision_recall(probs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    precision, recall, _ = metrics.precision_recall_curve(labels.numpy(), probs.numpy())
    pr_auc = metrics.auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    return pr_auc

def compute_f1_score(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).long()
    f1 = metrics.f1_score(labels.numpy(), preds.numpy())
    print(f"F1 Score (threshold={threshold}): {f1:.4f}")
    return f1

def compute_classification_report(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> str:
    preds = (probs >= threshold).long()
    report = metrics.classification_report(labels.numpy(), preds.numpy(), target_names=["Background", "Signal"])
    print(f"Classification Report (threshold={threshold}):\n{report}")
    return report

def plot_roc_curve(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str):
    fpr, tpr, _ = metrics.roc_curve(labels.numpy(), probs.numpy())
    roc_auc = metrics.auc(fpr, tpr)
    save_plot_data(bkg_type, "roc_curve", fpr=fpr, tpr=tpr, roc_auc=np.array([roc_auc]))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve for {bkg_type} Background")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_roc_curve.pdf")

def plot_precision_recall_curve(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str):
    precision, recall, _ = metrics.precision_recall_curve(labels.numpy(), probs.numpy())
    pr_auc = metrics.auc(recall, precision)
    save_plot_data(bkg_type, "precision_recall_curve",
                   precision=precision, recall=recall, pr_auc=np.array([pr_auc]))
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"Precision-Recall curve (AUC = {pr_auc:.4f})")
    plt.title(f"Precision-Recall Curve for {bkg_type} Background")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_precision_recall_curve.pdf")

def compute_confusion_matrix(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    preds = (probs >= threshold).long()
    cm = metrics.confusion_matrix(labels.numpy(), preds.numpy())
    print(f"Confusion Matrix (threshold={threshold}):\n{cm}")
    return cm

def efficiency_given_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> Tuple[float, float]:
    preds = (probs >= threshold).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    signal_efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    background_rejection = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
    return signal_efficiency, background_rejection

def efficiency_rejection_curves(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str) -> None:
    thresholds = np.linspace(0, 1, 100)
    signal_efficiencies = []
    background_rejections = []
    for thresh in thresholds:
        eff, rej = efficiency_given_threshold(probs, labels, thresh)
        signal_efficiencies.append(eff)
        background_rejections.append(rej)
    save_plot_data(bkg_type, "efficiency_rejection_curves",
                   signal_efficiencies=np.array(signal_efficiencies),
                   background_rejections=np.array(background_rejections))

    plt.figure(figsize=(6, 6))
    plt.plot(signal_efficiencies, background_rejections, label=f"Efficiency-Rejection curve")
    plt.title(f"Signal Efficiency vs Background Rejection \n {bkg_type} Background")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_efficiency_rejection_curve.pdf")

def efficiency_rejection_vs_threshold(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str) -> None:
    thresholds = np.linspace(0, 1, 100)
    signal_efficiencies = []
    background_rejections = []
    for thresh in thresholds:
        eff, rej = efficiency_given_threshold(probs, labels, thresh)
        signal_efficiencies.append(eff)
        background_rejections.append(rej)
    save_plot_data(bkg_type, "efficiency_rejection_vs_threshold",
                   thresholds=thresholds,
                   signal_efficiencies=np.array(signal_efficiencies),
                   background_rejections=np.array(background_rejections))

    plt.figure(figsize=(12, 5))
    
    plt.plot(thresholds, signal_efficiencies, label="Signal Efficiency", color="blue")
    plt.plot(thresholds, background_rejections, label="Background Rejection", color="red")
    plt.title(f"Efficiency and Rejection vs Threshold for {bkg_type} Background")
    plt.xlabel("Threshold") 
    plt.ylabel("Efficiency / Rejection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_efficiency_rejection_vs_threshold.pdf")


def plot_canvas(probs: torch.Tensor, labels: torch.Tensor, bkg_type: str) -> None:

    # --- save data ---
    fpr, tpr, _ = metrics.roc_curve(labels.numpy(), probs.numpy())
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(labels.numpy(), probs.numpy())
    pr_auc = metrics.auc(recall, precision)
    thresholds = np.linspace(0, 1, 100)
    signal_efficiencies = []
    background_rejections = []
    for thresh in thresholds:
        eff, rej = efficiency_given_threshold(probs, labels, thresh)
        signal_efficiencies.append(eff)
        background_rejections.append(rej)
    save_plot_data(bkg_type, "canvas",
                   probs=probs.numpy(), labels=labels.numpy(),
                   fpr=fpr, tpr=tpr, roc_auc=np.array([roc_auc]),
                   precision=precision, recall=recall, pr_auc=np.array([pr_auc]),
                   thresholds=thresholds,
                   signal_efficiencies=np.array(signal_efficiencies),
                   background_rejections=np.array(background_rejections))

    # --- plot ---
    fig = plt.figure(figsize=(12, 8))

    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1:], sharey=ax2)

    # prob distributions
    ax1.hist(probs[labels == 1].numpy(), bins=50, alpha=0.5,
             label="Signal", color="blue", density=True)
    ax1.hist(probs[labels == 0].numpy(), bins=50, alpha=0.5,
             label="Background", color="red", density=True)
    ax1.set_title(f"Predicted Probabilities for {bkg_type} Background")
    ax1.set_xlabel("Probability of Signal")
    ax1.set_ylabel("PDF")
    ax1.legend()
    ax1.grid(True)

    # roc
    ax2.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color='blue')
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_title(f"ROC Curve for {bkg_type} Background")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    ax2.grid(True)

    # efficiency vs threshold
    ax3.plot(thresholds, signal_efficiencies, label="Signal Efficiency", color="blue")
    ax3.plot(thresholds, background_rejections, label="Background Rejection", color="red")
    ax3.set_title(f"Efficiency and Rejection vs Threshold for {bkg_type} Background")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Efficiency / Rejection")
    ax3.legend()
    ax3.grid(True)

    fig.suptitle(f"Evaluation for {bkg_type} BKG")
    fig.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{bkg_type}_canvas.pdf")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bkg_type", default="pileup_MUON", choices=["pileup_MUON"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    loader = load_test_data(args.bkg_type, args.batch_size)
    model = load_model(args.bkg_type, args.checkpoint)
    probs, labels = run_inference(model, loader)

    n_sig = (labels == 1).sum().item()
    n_bkg = (labels == 0).sum().item()
    print(f"\nTest set: {n_sig} signal + {n_bkg} background = {n_sig + n_bkg} graphs")
    print(f"Probs:  shape={list(probs.shape)}, mean={probs.mean():.4f}")
    print(f"Labels: shape={list(labels.shape)}, sig_mean={probs[labels==1].mean():.4f}, "
          f"bkg_mean={probs[labels==0].mean():.4f}")

    ### SAVE RAW DATA ###

    save_plot_data(args.bkg_type, "raw",
                   probs=probs.numpy(), labels=labels.numpy())

    ### METRICS AND PLOTS ###

    plot_probability_distributions(probs, labels, args.bkg_type)
    compute_accuracy(probs, labels)
    compute_roc_auc(probs, labels)
    compute_precision_recall(probs, labels)
    compute_f1_score(probs, labels)
    compute_classification_report(probs, labels)
    plot_roc_curve(probs, labels, args.bkg_type)
    plot_precision_recall_curve(probs, labels, args.bkg_type)
    compute_confusion_matrix(probs, labels)
    efficiency_rejection_curves(probs, labels, args.bkg_type)
    efficiency_rejection_vs_threshold(probs, labels, args.bkg_type)


    # make a subplot for better visualization

    plot_canvas(probs, labels, args.bkg_type)
