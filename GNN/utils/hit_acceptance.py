import sys
import os
sys.path.append("/home3/alejandro.rodriguez/python_modules")

from functions import *
import numpy as np
from matplotlib.ticker import MaxNLocator

set_tfm_style()

# --- CODEX angular acceptance ---
theta_min = 0.8150
theta_max = 1.3104
phi_min   = -0.2608
phi_max   =  0.2608

eta_min = -np.log(np.tan(theta_max / 2))
eta_max = -np.log(np.tan(theta_min / 2))

# --- LOADING ---
DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"

SIG_IDS = ["40114060", "11114033"]
BKG_IDS = ["30011001", "38000800"]

BKG_FILES  = [f"{DATA_PATH}ntuple_background_{id}.root" for id in BKG_IDS]
SIG_FILES  = [f"{DATA_PATH}ntuple_signal_{id}.root" for id in SIG_IDS]

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

    df['rho'], df['eta'], df['phi'] = collider_system(df)

    df['in_codex'] = (df['eta'] >= eta_min) & (df['eta'] <= eta_max) & \
                     (df['phi'] >= phi_min) & (df['phi'] <= phi_max)

    dfs[i] = df

df_A0, df_phi, df_mu, df_kl0 = dfs

# --- per-event hits in CODEX acceptance ---
def count_in_codex(df):
    return df.groupby('eventNumber')['in_codex'].sum().values

combinaciones = [
    (df_A0, df_mu, 'Dark Photon', 'MUON'),
    (df_A0, df_kl0, 'Dark Photon', 'KL0'),
    (df_phi, df_mu, 'Dark Higgs', 'MUON'),
    (df_phi, df_kl0, 'Dark Higgs', 'KL0'),
]

bins = np.linspace(0, 50, 51)

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for idx, (sig, bkg, sig_label, bkg_label) in enumerate(combinaciones):
    ax = axes[idx]

    sig_counts = count_in_codex(sig)
    bkg_counts = count_in_codex(bkg)

    ax.hist(sig_counts, bins=bins, histtype='step', color='blue',
            label=sig_label, density=True, range=(0,50))
    ax.hist(bkg_counts, bins=bins, histtype='step', color='red',
            label=bkg_label, density=True, range=(0,50))

    ax.set_xlabel('Hits in acceptance per event')
    ax.set_ylabel('PDF')
    ax.set_yscale('log')
    ax.legend(fontsize=10)

fig.subplots_adjust(hspace=0.05, wspace=0.3, left=0.08, right=0.98, top=0.92, bottom=0.3)
fig.savefig('plots_tfm/hit_acceptance.pdf', bbox_inches='tight', pad_inches=0.5)
print('Saved plots_tfm/hit_acceptance.pdf')
