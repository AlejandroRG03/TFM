import sys
import os
import matplotlib.pyplot as plt

sys.path.append("/home3/alejandro.rodriguez/python_modules")
from functions import *

def set_tfm_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    })

set_tfm_style()

path = '/lustre/LHCb/alejandro.rodriguez/script_emilio_hits/ntuple_minbias_emilio.root'
tree = 'VeloMultiTuple_73eaa531/Clusters;1'
columns = ['eventNumber', 'x', 'y', 'z']
nrows = 10_000 

df = read_root(path, tree, columns, nrows)

fig, (ax1, ax2) = plt.subplots(
    1, 2, 
    figsize=(10, 4), 
    sharey=True, 
    gridspec_kw={'width_ratios': [1, 2], 'wspace': 0}
)
# Transverse hits
ax1.scatter(df['x'], df['y'], s=1, color='blue')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')

# Longitudinal hits
ax2.scatter(df['z'], df['y'], s=1, color='red')
ax2.set_xlabel('z [mm]')
#ax2.set_ylabel('x [mm]')

fig.tight_layout()
plt.savefig('first_plots/velo_shape.pdf', bbox_inches='tight')
plt.close()