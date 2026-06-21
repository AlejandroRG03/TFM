import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import sys
sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import set_tfm_style

set_tfm_style()

PLOTS_DATA_DIR = "test_data"


def list_available_datasets() -> list[str]:
    files = sorted(os.listdir(PLOTS_DATA_DIR))
    bkg_types = sorted(set(f.split("_")[0] for f in files if f.endswith(".npz")))
    for bkg in bkg_types:
        print(f"\n{bkg}:")
        for f in files:
            if f.startswith(bkg):
                print(f"  {f}")
    return bkg_types


def load_raw(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR, f"{bkg_type}_raw.npz")))


def load_probability_distributions(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR,
                                     f"{bkg_type}_probability_distributions.npz")))


def load_roc_curve(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR, f"{bkg_type}_roc_curve.npz")))


def load_precision_recall_curve(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR,
                                     f"{bkg_type}_precision_recall_curve.npz")))


def load_efficiency_rejection_curves(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR,
                                     f"{bkg_type}_efficiency_rejection_curves.npz")))


def load_efficiency_rejection_vs_threshold(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR,
                                     f"{bkg_type}_efficiency_rejection_vs_threshold.npz")))


def load_canvas(bkg_type: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(PLOTS_DATA_DIR, f"{bkg_type}_canvas.npz")))


if __name__ == "__main__":
    list_available_datasets()


prob_MUON = load_probability_distributions("MUON")
prob_KL0  = load_probability_distributions("KL0")

roc_MUON  = load_roc_curve("MUON")
roc_KL0   = load_roc_curve("KL0")

ef_MUON   = load_efficiency_rejection_vs_threshold("MUON")
ef_KL0    = load_efficiency_rejection_vs_threshold("KL0")

from matplotlib.ticker import MaxNLocator, MultipleLocator
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(2, 4, figure=fig, hspace=0, wspace=0.3)
gs_right = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 2:], wspace=0, hspace=0)

ax4 = fig.add_subplot(gs[1, :2])
ax1 = fig.add_subplot(gs[0, :2], sharex=ax4)
ax5 = fig.add_subplot(gs_right[1, 0])
ax6 = fig.add_subplot(gs_right[1, 1], sharey=ax5)
ax2 = fig.add_subplot(gs_right[0, 0], sharex=ax5, sharey=ax5)
ax3 = fig.add_subplot(gs_right[0, 1], sharex=ax6, sharey=ax6)

# Tick visibility
for ax in [ax1, ax2, ax3]:
    ax.tick_params(labelbottom=False)
for ax in [ax5, ax2]:
    ax.tick_params(labelleft=False)
for ax in [ax6, ax3]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

# Ticks fijos
fixed_ticks = [0.2, 0.4, 0.6, 0.8]
for ax in [ax2, ax5, ax3, ax6]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(fixed_ticks)
    ax.set_yticks(fixed_ticks)
for ax in [ax1, ax4]:
    ax.set_xlim(0, 1)
    ax.set_xticks(fixed_ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

# --- Probability ---
sig = prob_MUON["labels"] == 1
bkg = prob_MUON["labels"] == 0
ax1.hist(prob_MUON["probs"][sig], bins=50, alpha=0.5, label="Dark Photon", color="blue", density=True)
ax1.hist(prob_MUON["probs"][bkg], bins=50, alpha=0.5, label="MUON",   color="red",  density=True)
ax1.set_ylabel("PDF")
ax1.legend(loc="upper left")
ax1.set_title("Probability distributions")

sig = prob_KL0["labels"] == 1
bkg = prob_KL0["labels"] == 0
ax4.hist(prob_KL0["probs"][sig], bins=50, alpha=0.5, label="Dark Photon", color="blue", density=True)
ax4.hist(prob_KL0["probs"][bkg], bins=50, alpha=0.5, label="KL0",    color="red",  density=True)
ax4.set_xlabel("Probability of Signal")
ax4.set_ylabel("PDF")
ax4.legend(loc="upper left")

# --- ROC ---
ax2.plot(roc_MUON["fpr"], roc_MUON["tpr"], color="blue")
ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax2.set_ylabel("TPR")
ax2.set_title("ROC")
ax2.annotate(f"MUON\nAUC = {roc_MUON['roc_auc'][0]:.4f}",
             xy=(0.97, 0.05), xycoords="axes fraction",
             fontsize=12, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax5.plot(roc_KL0["fpr"], roc_KL0["tpr"], color="blue")
ax5.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax5.set_xlabel("FPR")
ax5.set_ylabel("TPR")
ax5.annotate(f"KL0\nAUC = {roc_KL0['roc_auc'][0]:.4f}",
             xy=(0.97, 0.05), xycoords="axes fraction",
             fontsize=12, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# --- Efficiency ---
ax3.plot(ef_MUON["thresholds"], ef_MUON["signal_efficiencies"],   color="blue")
ax3.plot(ef_MUON["thresholds"], ef_MUON["background_rejections"], color="red")
ax3.set_title("Efficiency / Rejection")
ax3.annotate("Signal eff.", xy=(1.2, 0.5), xycoords="axes fraction",
             color="blue", fontsize=15, rotation=90, va="center", ha="left")
ax3.annotate("Bkg. rej.",   xy=(1.3, 0.5), xycoords="axes fraction",
             color="red",   fontsize=15, rotation=90, va="center", ha="left")
ax3.annotate("MUON", xy=(0.4, 0.05), xycoords="axes fraction",
             fontsize=12, ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax6.plot(ef_KL0["thresholds"], ef_KL0["signal_efficiencies"],   color="blue")
ax6.plot(ef_KL0["thresholds"], ef_KL0["background_rejections"], color="red")
ax6.set_xlabel("Threshold")
ax6.annotate("Signal eff.", xy=(1.2, 0.5), xycoords="axes fraction",
             color="blue", fontsize=15, rotation=90, va="center", ha="left")
ax6.annotate("Bkg. rej.",   xy=(1.3, 0.5), xycoords="axes fraction",
             color="red",   fontsize=15, rotation=90, va="center", ha="left")
ax6.annotate("KL0", xy=(0.43, 0.05), xycoords="axes fraction",
             fontsize=12, ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.savefig("test_plots/ALL.pdf", bbox_inches="tight")