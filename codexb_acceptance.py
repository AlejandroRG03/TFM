from functions import *

# Load data

path = '/scratch48/emilio.fernandez/Velo/VELOHits.root'
tree = 'VeloMultiTuple_73eaa531/Clusters;1'
columns = ['eventNumber', 'x', 'y', 'z']
nrows = 100000 # load just the fist 100 000 of rows

df = read_root(path, tree, columns, nrows=nrows)

# Define CODEX-B acceptance geometry, model the acceptance as a cone of radius d at distance r from the interaction point

d = 10000 # radius of the cone

x = 23725
y = 0
z = 12650

r = np.sqrt( x**2 + y**2 + z**2 )

max_angle = np.arctan(d/r)

codexb_axis = np.array([x, y, z])
codexb_axis = codexb_axis / np.linalg.norm(codexb_axis)

# Compute angles

df['angles'] = compute_angles(codexb_axis, df)

# Check which hits are within the acceptance

df['in_codexb'] = df['angles'].abs() < max_angle

# Plot hits within acceptance

plt.figure()

plt.plot(df[df['in_codexb']]['z'], df[df['in_codexb']]['x'], '.', color = 'green', zorder=1, label='In CODEX-B acceptance')
plt.plot(df[~df['in_codexb']]['z'], df[~df['in_codexb']]['x'], '.', color = 'blue', zorder=0, label='Out of CODEX-B acceptance')

# plot the cone boundary

a_z = codexb_axis[2] * (np.cos(max_angle) - np.sin(max_angle)) * 70
a_x = codexb_axis[0] * (np.cos(max_angle) + np.sin(max_angle)) * 70

b_z = codexb_axis[2] * (np.cos(max_angle) + np.sin(max_angle)) * 150
b_x = codexb_axis[0] * (np.cos(max_angle) - np.sin(max_angle)) * 150

plt.plot([0, a_z], [0, a_x], 'r--', label='CODEX-B acceptance boundary', zorder=2)
plt.plot([0, b_z], [0, b_x], 'r--', zorder=2)


plt.grid(linestyle='--')
plt.xlabel('z')
plt.ylabel('x')
plt.legend()

plt.savefig('first_plots/codexb_acceptance.png')
plt.close()