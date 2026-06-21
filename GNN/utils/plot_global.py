import sys
import os
sys.path.append("/home3/alejandro.rodriguez/python_modules")

from functions import *
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set plotting style
set_tfm_style()

# --- define codex region ---
CODEX_X = 23725  # mm
CODEX_Y = 0      # mm
CODEX_Z = 12650  # mm
CODEX_L = 10_000 # mm

CODEX_CENTER = np.array([CODEX_X, CODEX_Y, CODEX_Z])
CODEX_AXIS = CODEX_CENTER / np.linalg.norm(CODEX_CENTER)

X_MIN, X_MAX = CODEX_X - CODEX_L/2, CODEX_X + CODEX_L/2
Y_MIN, Y_MAX = CODEX_Y - CODEX_L/2, CODEX_Y + CODEX_L/2
Z_MIN, Z_MAX = CODEX_Z - CODEX_L/2, CODEX_Z + CODEX_L/2

# collider system transformation of codex

theta_min = 0.8150 # rad
theta_max = 1.3104 # rad
phi_min   = -0.2608 # rad
phi_max   = 0.2608 # rad

eta_min = -np.log(np.tan(theta_max/2))
eta_max = -np.log(np.tan(theta_min/2))


# LOADING

DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"

SIG_IDS = ["40114060", "11114033"]
BKG_IDS    = ["30011001", "38000800"]

BKG_FILES  =  [f"{DATA_PATH}ntuple_background_{id}.root" for id in BKG_IDS]
SIG_FILES = [f"{DATA_PATH}ntuple_signal_{id}.root" for id in SIG_IDS]

LABELS = {
    "40114060" : r"SIG: $H^0\to 2 A^0$",
    "11114033" : r"SIG: $B^0 \to K^{*0}\phi$",
    "30011001" : r"BKG: MUON",
    "38000800" : r"BKG: KL0",
}

VAR_NAMES = ['bxType', 'eventNumber', 'bxId', 'gpsTime', 'runNumber', 'triggerType', 'eventType', 
             'beamspotX', 'beamspotY', 'nTrk_per_event', 'nVtx_per_event', 'nClu_per_event', 
             'module', 'chip', 'sensor', 'row', 'col', 'n_pix', 'x', 'y', 'z']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"


print('Reading data...')
bkg_df = [read_root(file, TREE_NAME, VAR_NAMES, nrows=20_000_000) for file in BKG_FILES]
sig_df = [read_root(file, TREE_NAME, VAR_NAMES, nrows=20_000_000) for file in SIG_FILES]
print('Data loaded!')


df_A0  = sig_df[0]
df_phi = sig_df[1]
df_mu  = bkg_df[0]
df_kl0 = bkg_df[1]

dfs = [df_A0, df_phi, df_mu, df_kl0]

# --- feature engineering ---

for i, df in enumerate(dfs):

    last_event = df['eventNumber'].unique()[-1]
    df = df[df['eventNumber'] != last_event].copy()

    df['x'] = df['x'] - df['beamspotX']
    df['y'] = df['y'] - df['beamspotY']

    # (x, y, z) -> (rho, eta, phi)
    df['rho'], df['eta'], df['phi'] = collider_system(df)

    # compute codex angles
    df['codex_angle'] = compute_angles(CODEX_AXIS, df)


    # in_codex condition
    df['in_codex'] = (df['eta'] >= eta_min) & (df['eta'] <= eta_max) & (df['phi'] >= phi_min) & (df['phi'] <= phi_max)

    # rename columns for plotting convenience
    df = df.rename(columns={'nTrk_per_event': 'nTrk',
                            'nVtx_per_event': 'nVtx',
                            'nClu_per_event': 'nClu'})

    # total hits per event
    total_hits_per_event = df.groupby('eventNumber').size()
    df['total_hits'] = df['eventNumber'].map(total_hits_per_event)

    dfs[i] = df
# reassign from updated list so standalone variables reflect renames
df_A0, df_phi, df_mu, df_kl0 = dfs

from matplotlib.ticker import ScalarFormatter, MaxNLocator

def plot_global(ax, bkg, sig, bkg_label, sig_label, col):
    if col == 'nVtx':
        vmax = int(max(bkg[col].max(), sig[col].max()))
        bin_edges = np.arange(-0.5, vmax + 1.5, 1)
    else:
        vmin = min(bkg[col].min(), sig[col].min())
        vmax = max(bkg[col].max(), sig[col].max())
        bin_edges = np.linspace(vmin, vmax, 31)  # 30 bins, edges compartidos

    ax.hist(sig[col], histtype='step', color='blue', label=sig_label, density=1, bins=bin_edges)
    ax.hist(bkg[col], histtype='step', color='red', label=bkg_label, density=1, bins=bin_edges)
    ax.set_xlabel(col if col != 'total_hits' else 'Hits per event')
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    ymax = ax.get_ylim()[1]
    exp = int(np.floor(np.log10(ymax))) if ymax > 0 else 0

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((exp, exp))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_ylabel(f'PDF ($\\times 10^{{{exp}}}$)')


variables = ['nTrk', 'nVtx', 'nClu']
combinaciones = [
    (df_A0, df_mu, 'Dark Photon', 'MUON'),
    (df_A0, df_kl0, 'Dark Photon', 'KL0'),
    (df_phi, df_mu, 'Dark Higgs', 'MUON'),
    (df_phi, df_kl0, 'Dark Higgs', 'KL0'),
]

fig, axes = plt.subplots(3, 4, figsize=(16, 9))

for row_idx, var in enumerate(variables):
    for col_idx, (sig, bkg, sig_label, bkg_label) in enumerate(combinaciones):
        ax = axes[row_idx, col_idx]
        plot_global(ax, bkg, sig, bkg_label, sig_label, var)

fig.subplots_adjust(hspace=0.4, wspace=0.3, left=0.06, right=0.98, top=0.96, bottom=0.06)
fig.savefig('plots_tfm/global.pdf', bbox_inches='tight')