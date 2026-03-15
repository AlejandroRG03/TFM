from functions import *

path = '/home3/alejandro.rodriguez/DecFiles/DVntuple.root'
tree = 'MCDecayTreeTuple/MCDecayTree;1'
columns = [
    "KL0_TRUEP_E",
    "KL0_TRUEP_X",
    "KL0_TRUEP_Y",
    "KL0_TRUEP_Z",
    "KL0_TRUEPT",
]


df = read_root(path, tree, columns)

print(df.shape)

# Define CODEX-B acceptance geometry, model the acceptance as a cone of radius d at distance r from the interaction point

r = 10000 # radius of the cone in mm

x = 23725 # mm
y = 0 # mm
z = 12650 # mm

d = np.sqrt( x**2 + y**2 + z**2 )

codex_angular_aperture = np.arctan(r/d)

# check if the KL0 particles are within the acceptance

codexb_center = np.array([x, y, z])
codexb_axis = codexb_center / np.linalg.norm(codexb_center)

# Compute min theta and max theta for codexb

theta_codexb = np.arccos(codexb_center[2] / np.linalg.norm(codexb_center))


df['x'] = df['KL0_TRUEP_X']
df['y'] = df['KL0_TRUEP_Y']
df['z'] = df['KL0_TRUEP_Z']

df['codex_angles'] = compute_angles(codexb_axis, df)

df['in_codex'] = df['codex_angles'] < codex_angular_aperture
df_in_codex = df[df['in_codex']]
df_not_in_codex = df[~df['in_codex']]

print(df.head())

print(f'Number of KL0 particles in CODEX-B acceptance: {df_in_codex.shape[0]}')
print(f'Number of KL0 particles out of CODEX-B acceptance: {df_not_in_codex.shape[0]}')
print(f'Total KL0 particles: {df.shape[0]}')
print(f'KL0_in/KL0_total: {df_in_codex.shape[0] / df.shape[0]}')

# histogram of angles

plt.figure()
plt.hist(df['codex_angles'] * 180 / np.pi, bins=50, color='blue', alpha=0.7, label='All KL0 particles')
plt.axvline(codex_angular_aperture * 180 / np.pi, color='red', linestyle='--', label='CODEX-B acceptance boundary')
plt.xlabel('Angle with respect to CODEX-B axis (degrees)')
plt.ylabel('Number of KL0 particles')
plt.legend()
plt.grid(linestyle='--')

plt.savefig('first_plots/KL0_codex_angles_histogram.png')