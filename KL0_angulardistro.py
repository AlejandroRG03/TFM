from functions import *

pathKL0 = '/home3/alejandro.rodriguez/DecFiles/KL0ntuple.root'
pathmup = '/home3/alejandro.rodriguez/DecFiles/mupntuple.root'
pathmun = '/home3/alejandro.rodriguez/DecFiles/mumntuple.root'
tree = 'MCDecayTreeTuple/MCDecayTree;1'
columnsKL0 = [
    "KL0_TRUEP_E",
    "KL0_TRUEP_X",
    "KL0_TRUEP_Y",
    "KL0_TRUEP_Z",
    "KL0_TRUEPT",
]
columnsmum = [  # gaudi saves mu+ as mu-, so same names
    "muminus_TRUEP_E",
    "muminus_TRUEP_X",
    "muminus_TRUEP_Y",
    "muminus_TRUEP_Z",
    "muminus_TRUEPT",
]


df_KL0 = read_root(pathKL0, tree=tree, columns=columnsKL0)
df_mup = read_root(pathmup, tree=tree, columns=columnsmum)
df_mum = read_root(pathmun, tree=tree, columns=columnsmum)

# remove the particle identifier from the column names 
df_KL0.columns = [col.replace('KL0_', '') for col in df_KL0.columns]
df_mup.columns = [col.replace('muminus_', '') for col in df_mup.columns]
df_mum.columns = [col.replace('muminus_', '') for col in df_mum.columns]

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

dfs = [df_KL0, df_mup, df_mum]

for df in dfs:

    df['x'] = df['TRUEP_X']
    df['y'] = df['TRUEP_Y']
    df['z'] = df['TRUEP_Z']

    df['codex_angles'] = compute_angles(codexb_axis, df)
    df['in_codex'] = df['codex_angles'] < codex_angular_aperture


print(f'Number of KL0 particles in CODEX-b acceptance: {df_KL0[df_KL0["in_codex"]].shape[0]}')
print(f'Number of KL0 particles out of CODEX-b acceptance: {df_KL0[~df_KL0["in_codex"]].shape[0]}')
print(f'Total KL0 particles: {df_KL0.shape[0]}')
print(f'KL0_in/KL0_total: {df_KL0[df_KL0["in_codex"]].shape[0] / df_KL0.shape[0]}')

print(f'Number of mu+ particles in CODEX-b acceptance: {df_mup[df_mup["in_codex"]].shape[0]}')
print(f'Number of mu- particles in CODEX-b acceptance: {df_mum[df_mum["in_codex"]].shape[0]}')
print(f'Total mu+ particles: {df_mup.shape[0]}')
print(f'Total mu- particles: {df_mum.shape[0]}')
print(f'mu+_in/mu+_total: {df_mup[df_mup["in_codex"]].shape[0] / df_mup.shape[0]}')
print(f'mu-_in/mu-_total: {df_mum[df_mum["in_codex"]].shape[0] / df_mum.shape[0]}')

# histogram of angles

# KL0
plt.figure(figsize=(10, 6))
plt.hist(df_KL0['codex_angles'], bins=50, range=(0, np.pi/2), alpha=0.7, label='KL0')
plt.axvline(x=codex_angular_aperture, color='r', linestyle='--', label='CODEX-b acceptance')
plt.xlabel('Angle with respect to CODEX-b axis (radians)')
plt.ylabel('Number of particles')
plt.title('Angular distribution of KL0 particles')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('first_plots/KL0_angular_distribution.png')

# mu+
plt.figure(figsize=(10, 6))
plt.hist(df_mup['codex_angles'], bins=50, range=(0, np.pi/2), alpha=0.7, label='mu+')
plt.axvline(x=codex_angular_aperture, color='r', linestyle='--', label='CODEX-b acceptance')
plt.xlabel('Angle with respect to CODEX-b axis (radians)')
plt.ylabel('Number of particles')
plt.title('Angular distribution of mu+ particles')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('first_plots/muplus_angular_distribution.png')

# mu-
plt.figure(figsize=(10, 6))
plt.hist(df_mum['codex_angles'], bins=50, range=(0, np.pi/2), alpha=0.7, label='mu-')
plt.axvline(x=codex_angular_aperture, color='r', linestyle='--', label='CODEX-b acceptance')
plt.xlabel('Angle with respect to CODEX-b axis (radians)')
plt.ylabel('Number of particles')
plt.title('Angular distribution of mu- particles')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('first_plots/muminus_angular_distribution.png')