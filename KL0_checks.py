from functions import *

pathKL0 = '/home3/alejandro.rodriguez/DecFiles/KL0_try.root'
pathMu  = '/home3/alejandro.rodriguez/DecFiles/Mu_try.root'
tree = 'MCDecayTreeTuple/MCDecayTree;1'
columnsKL0 = [
    "KL0_TRUEP_E",
    "KL0_TRUEP_X",
    "KL0_TRUEP_Y",
    "KL0_TRUEP_Z",
    "KL0_TRUEPT",
    "KL0_TRUEORIGINVERTEX_X",
    "KL0_TRUEORIGINVERTEX_Y",
    "KL0_TRUEORIGINVERTEX_Z",
    "EventInSequence"
]

columnsMu = [
    "muminus_TRUEP_E",
    "muminus_TRUEP_X",
    "muminus_TRUEP_Y",
    "muminus_TRUEP_Z",
    "muminus_TRUEPT",
    "muminus_TRUEORIGINVERTEX_X",
    "muminus_TRUEORIGINVERTEX_Y",
    "muminus_TRUEORIGINVERTEX_Z",
    "EventInSequence",
    "muminus_MC_MOTHER_ID"
]

df_KL0 = read_root(pathKL0, tree=tree, columns=columnsKL0)
df_Mu  = read_root(pathMu, tree=tree, columns=columnsMu)

# remove the particle identifier from the column names
df_KL0.columns = [col.replace('KL0_', '') for col in df_KL0.columns]
df_Mu.columns = [col.replace('muminus_', '') for col in df_Mu.columns]

print(df_KL0.head())
print(df_Mu.head())

# --- VELO acceptance geometry ---

rho_max    = 42.0 # mm

VELO_z_min = -287.4 # mm
VELO_z_max = 751.0 # mm


# --- CODEX-B acceptance geometry ---

df_KL0['theta'], df_KL0['phi'] = spherical_angles(df_KL0, x='TRUEP_X', y='TRUEP_Y', z='TRUEP_Z')
df_Mu['theta'], df_Mu['phi'] = spherical_angles(df_Mu, x='TRUEP_X', y='TRUEP_Y', z='TRUEP_Z')

theta_min = 0.8150  # rad
theta_max = 1.3104  # rad
phi_min   = -0.2608 # rad
phi_max   = 0.2608  # rad

df_KL0['in_velo'] =  (df_KL0['TRUEORIGINVERTEX_X']**2 + df_KL0['TRUEORIGINVERTEX_Y']**2 < rho_max**2) & \
                     (df_KL0['TRUEORIGINVERTEX_Z'] < VELO_z_max) & (df_KL0['TRUEORIGINVERTEX_Z'] > VELO_z_min) # True if the particle originates within the VELO acceptance


df_Mu['in_velo'] =  (df_Mu['TRUEORIGINVERTEX_X']**2 + df_Mu['TRUEORIGINVERTEX_Y']**2 < rho_max**2) & \
                     (df_Mu['TRUEORIGINVERTEX_Z'] < VELO_z_max) & (df_Mu['TRUEORIGINVERTEX_Z'] > VELO_z_min) # True if the particle originates within the VELO acceptance

df_KL0['in_codex'] = ((df_KL0['theta'] > theta_min) & (df_KL0['theta'] < theta_max) & (df_KL0['phi'] > phi_min) & (df_KL0['phi'] < phi_max))
df_Mu['in_codex'] = ((df_Mu['theta'] > theta_min) & (df_Mu['theta'] < theta_max) & (df_Mu['phi'] > phi_min) & (df_Mu['phi'] < phi_max))


df_KL0['isGood'] = df_KL0['in_velo'] & df_KL0['in_codex']
df_Mu['isGood'] = df_Mu['in_velo'] & df_Mu['in_codex']


# --- print results ---

print("Total KL0 particles: ", len(df_KL0))
print("Total events: ", df_KL0['EventInSequence'].nunique())
print("KL0 particles in VELO acceptance: ", df_KL0['in_velo'].sum())
print("KL0 particles in CODEX-B acceptance: ", df_KL0['in_codex'].sum())
print("KL0 particles in both VELO and CODEX-B acceptance: ", df_KL0['isGood'].sum())

print("Total Mu particles: ", len(df_Mu))
print("Total events: ", df_Mu['EventInSequence'].nunique())
print("Mu particles in VELO acceptance: ", df_Mu['in_velo'].sum())
print("Mu particles in CODEX-B acceptance: ", df_Mu['in_codex'].sum())
print("Mu particles in both VELO and CODEX-B acceptance: ", df_Mu['isGood'].sum())

# Mu mother ID

KL0_ID = 130
df_Mu['from_KL0'] = df_Mu[df_Mu['isGood']]['MC_MOTHER_ID'] == KL0_ID
print("Mu particles from KL0 decays verifying conditions: ", df_Mu['from_KL0'].sum())
print("Mu good mother IDs: ", df_Mu[df_Mu['isGood']]['MC_MOTHER_ID'].value_counts())

# check if all events have at least one good KL0

events_with_good_KL0 = df_KL0[df_KL0['isGood']]['EventInSequence'].unique()
print("Events with at least one good KL0: ", events_with_good_KL0)

events_with_good_Mu = df_Mu[df_Mu['isGood']]['EventInSequence'].unique()
print("Events with at least one good Mu: ", events_with_good_Mu)