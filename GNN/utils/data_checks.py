import sys
import os
sys.path.append("/home3/alejandro.rodriguez/python_modules")

from functions import *

# Set plotting style
set_tfm_style()

"""
This script goal is to check if our background and signal datasets are separable, and to check
the main differences in the data.
"""

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

# --- load the data ---
DATA_PATH = "/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/"
BKG_LABEL = "MUON"


DEC_ID    = "38000800" if BKG_LABEL == "KL0" else "30011001"
BKG_FILE = f"{DATA_PATH}ntuple_background_{DEC_ID}.root"
SIG_FILE = f"{DATA_PATH}ntuple_signal_40114060.root"

VAR_NAMES = ['bxType', 'eventNumber', 'bxId', 'gpsTime', 'runNumber', 'triggerType', 'eventType', 
             'beamspotX', 'beamspotY', 'nTrk_per_event', 'nVtx_per_event', 'nClu_per_event', 
             'module', 'chip', 'sensor', 'row', 'col', 'n_pix', 'x', 'y', 'z']
TREE_NAME = "VeloMultiTuple_73eaa531/Clusters"

print('Reading data...')
bkg_df = read_root(BKG_FILE, TREE_NAME, VAR_NAMES, nrows=10_000_000)
sig_df = read_root(SIG_FILE, TREE_NAME, VAR_NAMES, nrows=10_000_000)
print('Data loaded!')

# drop the rows with the last eventNumber since it is incomplete
last_event_bkg = bkg_df['eventNumber'].iloc[-1]
last_event_sig = sig_df['eventNumber'].iloc[-1]

bkg_df = bkg_df[bkg_df['eventNumber'] != last_event_bkg]
sig_df = sig_df[sig_df['eventNumber'] != last_event_sig]

# --- feature engineering ---

# (x, y, z) -> (rho, eta, phi)
bkg_df['rho'], bkg_df['eta'], bkg_df['phi'] = collider_system(bkg_df)
sig_df['rho'], sig_df['eta'], sig_df['phi'] = collider_system(sig_df)

# compute codex angles
bkg_df['codex_angle'] = compute_angles(CODEX_AXIS, bkg_df)
sig_df['codex_angle'] = compute_angles(CODEX_AXIS, sig_df)

# in_codex condition
bkg_df['in_codex'] = (bkg_df['eta'] >= eta_min) & (bkg_df['eta'] <= eta_max) & (bkg_df['phi'] >= phi_min) & (bkg_df['phi'] <= phi_max)
sig_df['in_codex'] = (sig_df['eta'] >= eta_min) & (sig_df['eta'] <= eta_max) & (sig_df['phi'] >= phi_min) & (sig_df['phi'] <= phi_max)

# global event variables

def aligned_hits_per_event(df_group, eta_width, phi_width):
    """
    Computes the number of "aligned" hits per event in the CODEX region
    """
    if df_group.empty:
        return 0
    counts, bins = np.histogramdd(df_group[['eta', 'phi']].values, bins=[np.arange(eta_min, eta_max+eta_width, eta_width), np.arange(phi_min, phi_max+phi_width, phi_width)])
    return (counts > 1).sum()  # Count bins with more than 2 hits



def extract_event_features(df):
    """
    Extracts event-level features by grouping hit-level data by eventNumber.

    :param df: DataFrame containing hit-level information and engineered features.
    :return: DataFrame with one row per event containing aggregated features.
    """
    grouped = df.groupby('eventNumber')
    df_codex = df[df['in_codex']]
    grouped_codex = df_codex.groupby('eventNumber')

    eta_width = 0.01
    phi_width = 0.01  # ~10 mrad

    return pd.DataFrame({
        'nTrk': grouped['nTrk_per_event'].first(),
        'nVtx': grouped['nVtx_per_event'].first(),
        'nClu': grouped['nClu_per_event'].first(),
        'total_hits': grouped.size(),
        'hits_in_codex': grouped['in_codex'].sum(),
        'fraction_hits_in_codex': grouped['in_codex'].mean(),
        'mean_codex_angle_per_event': grouped['codex_angle'].mean(),

        'sum_npix_codex': grouped_codex['n_pix'].sum(),
        'mean_npix_codex': grouped_codex['n_pix'].mean(),
        'max_npix_codex': grouped_codex['n_pix'].max(),
        
        # Qué tan "juntos" están los hits (proxy de si forman una traza)
        'std_eta_codex': grouped_codex['eta'].std(),
        'std_phi_codex': grouped_codex['phi'].std(),

        # aligned hits per event in codex
        'aligned_hits_codex': grouped_codex.apply(lambda g: aligned_hits_per_event(g, eta_width=eta_width, phi_width=phi_width)),
        'aligned_hits_fraction_codex': grouped_codex.apply(lambda g: aligned_hits_per_event(g, eta_width=eta_width, phi_width=phi_width) / len(g) if len(g) > 0 else 0)
    })

bkg_event_df = extract_event_features(bkg_df)
sig_event_df = extract_event_features(sig_df)

# --- plot definition ---

def plot_eta_phi(bkg, sig, zoomed=False):
    """
    Plots 2D histograms of eta vs phi for background and signal, along with their residuals.

    :param bkg: DataFrame containing background hits.
    :param sig: DataFrame containing signal hits.
    :param zoomed: Boolean, if True, crops the plot to the CODEX-B acceptance region.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    
    range_lims = [[eta_min, eta_max], [phi_min, phi_max]] if zoomed else None
    bins = 50 if zoomed else 100

    # Background & Signal 2D
    ax1.hist2d(bkg['eta'], bkg['phi'], bins=bins, cmap='viridis', range=range_lims)
    ax2.hist2d(sig['eta'], sig['phi'], bins=bins, cmap='viridis', range=range_lims)
    
    # Residuals
    h_bkg, x_e, y_e = np.histogram2d(bkg['eta'], bkg['phi'], bins=bins, range=range_lims)
    h_sig, _, _ = np.histogram2d(sig['eta'], sig['phi'], bins=[x_e, y_e])
    pcm = ax3.pcolormesh(x_e, y_e, (h_sig - h_bkg).T, cmap='seismic', shading='auto')

    for ax, title in zip([ax1, ax2, ax3], [f'Background ({BKG_LABEL})', 'Signal', 'Residuals (Sig-Bkg)']):
        ax.set_title(f"{title} (eta, phi) {'- Zoomed' if zoomed else ''}")
        ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$\phi$')
        if not zoomed:
            ax.add_patch(plt.Rectangle((eta_min, phi_min), eta_max-eta_min, phi_max-phi_min, 
                                       fill=False, edgecolor='red', linewidth=2, label='Codex'))
            ax.legend()

    fig.colorbar(pcm, ax=ax3, label='Residuals')
    plt.tight_layout()
    suffix = "_zoomed" if zoomed else ""
    plt.savefig(f"check_plots/eta_phi{suffix}.png")
    plt.close()


# Nos aseguramos de que el directorio de salida existe
os.makedirs("check_plots", exist_ok=True)

plot_eta_phi(bkg_df, sig_df, zoomed=False)
plot_eta_phi(bkg_df, sig_df, zoomed=True)

def plot_1d_comparison(bkg_data, sig_data, title, xlabel, filename, density=True, logx=False, logy=False, bins=100, window=None, is_discrete=False):
    """
    Plots a 1D histogram comparison between background and signal datasets.

    :param bkg_data: Series or array containing background values.
    :param sig_data: Series or array containing signal values.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param filename: Name of the file to save the plot.
    :param density: Boolean, if True, normalizes the histograms to form a probability density.
    :param is_discrete: Boolean, if True, forces bins to perfectly frame integer values.
    """
    plt.figure(figsize=(6, 5))
    
    # Si la variable es un conteo (discreta), forzamos los bines
    if is_discrete:
        min_val = 0 if window is None else int(window[0])
        max_val = int(max(bkg_data.max(), sig_data.max())) if window is None else int(window[1])
        
        # Creamos los bordes de los bines: [-0.5, 0.5, 1.5, 2.5 ... max_val + 0.5]
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    # Ploteamos con histtype='step' normal, pero ahora 'bins' puede ser el array que acabamos de crear
    plt.hist(bkg_data, bins=bins, histtype='step', color='red', label=f'Background ({BKG_LABEL})', density=density, range=window, linewidth=1.5)
    plt.hist(sig_data, bins=bins, histtype='step', color='blue', label='Signal (LLP)', density=density, range=window, linewidth=1.5)
    
    if logx: plt.xscale('log')
    if logy: plt.yscale('log')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density' if density else 'Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"check_plots/{filename}")
    plt.close()


# --- plots  ---

plot_1d_comparison(bkg_df['codex_angle'], sig_df['codex_angle'], 
                   'Codex Angles', 'Angle (rad)', 'codex_angle.png')
plot_1d_comparison(bkg_event_df['mean_codex_angle_per_event'], sig_event_df['mean_codex_angle_per_event'], 
                   'Mean Codex Angles per Event', 'Angle (rad)', 'codex_angle_per_event.png')
plot_1d_comparison(bkg_event_df['hits_in_codex'], sig_event_df['hits_in_codex'], 
                   'Hits in Codex per Event', 'Number of hits', 'hits_in_codex_per_event.png',
                    logy=True, window=(0, 30), is_discrete=True)
# multiplicity per event
plot_1d_comparison(bkg_event_df['total_hits'], sig_event_df['total_hits'], 
                   'Total VELO Hits per Event', 'Hits', 'total_hits_per_event.png',
                   density=True, is_discrete=False, window=(0, 5000)) # Ajusta la ventana
# fraction of hits in codex per event
plot_1d_comparison(bkg_event_df['fraction_hits_in_codex'].dropna(), sig_event_df['fraction_hits_in_codex'].dropna(), 
                   'Fraction of Hits in Codex', 'Fraction', 'fraction_in_codex.png',
                   density=True, logy=True, bins=50, window=(0, 0.05))


mask_bkg = bkg_event_df['hits_in_codex'] > 1
mask_sig = sig_event_df['hits_in_codex'] > 1

# sum of n_pix in codex per event
plot_1d_comparison(bkg_event_df[mask_bkg]['sum_npix_codex'], sig_event_df[mask_sig]['sum_npix_codex'], 
                   'Total n_pix deposited in Codex region per Event', 'Sum of n_pix', 'sum_npix_codex.png',
                   density=True, is_discrete=True, window=(0, 50))

plot_1d_comparison(bkg_event_df[mask_bkg]['mean_npix_codex'], sig_event_df[mask_sig]['mean_npix_codex'], 
                   'Mean n_pix deposited in Codex region per Event', 'Mean n_pix', 'mean_npix_codex.png',
                   density=True, is_discrete=True, window=(0, 10))

plot_1d_comparison(bkg_event_df[mask_bkg]['max_npix_codex'], sig_event_df[mask_sig]['max_npix_codex'], 
                   'Max n_pix deposited in Codex region per Event', 'Max n_pix', 'max_npix_codex.png',
                   density=True, is_discrete=True, window=(0, 20))

# angular dispersion of hits in codex
plot_1d_comparison(bkg_event_df[mask_bkg]['std_eta_codex'], sig_event_df[mask_sig]['std_eta_codex'], 
                   'Spread (Std) of Eta for hits per Event in Codex', 'Std Eta', 'std_eta_codex.png',
                   density=True, bins=50, window=(0, 0.5))

plot_1d_comparison(bkg_event_df[mask_bkg]['std_phi_codex'], sig_event_df[mask_sig]['std_phi_codex'], 
                   'Spread (Std) of Phi for hits per Event in Codex', 'Std Phi', 'std_phi_codex.png',
                   density=True, bins=50, window=(0, 0.5))

# nTrk_per_event', 'nVtx_per_event', 'nClu_per_event'
plot_1d_comparison(bkg_event_df['nTrk'], sig_event_df['nTrk'], 
                   'Number of Tracks per Event', 'nTrk', 'nTrk_per_event.png',
                   density=True, window=(0, 500))
plot_1d_comparison(bkg_event_df['nVtx'], sig_event_df['nVtx'], 
                   'Number of Vertices per Event', 'nVtx', 'nVtx_per_event.png',
                   density=True, window=(0, 20))
plot_1d_comparison(bkg_event_df['nClu'], sig_event_df['nClu'], 
                   'Number of Clusters per Event', 'nClu', 'nClu_per_event.png',
                   density=True, window=(0, 5000))

# aligned hits in codex per event
plot_1d_comparison(bkg_event_df[mask_bkg]['aligned_hits_codex'], sig_event_df[mask_sig]['aligned_hits_codex'], 
                   'Number of Aligned Hits in Codex per Event', 'Aligned Hits', 'aligned_hits_codex.png',
                   density=True, is_discrete=True, window=(0, 10), logy=True)

plot_1d_comparison(bkg_event_df[mask_bkg]['aligned_hits_fraction_codex'], sig_event_df[mask_sig]['aligned_hits_fraction_codex'], 
                   'Fraction of Aligned Hits in Codex per Event', 'Aligned Hits Fraction', 'aligned_hits_fraction_codex.png',
                   density=True, bins=20, window=(0, 1), logy=True) # of hits in codex, how many are aligned?