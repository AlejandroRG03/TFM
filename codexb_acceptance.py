from functions import *

# Load data

path = '/scratch48/emilio.fernandez/Velo/VELOHits.root'
tree = 'VeloMultiTuple_73eaa531/Clusters;1'
columns = ['eventNumber', 'x', 'y', 'z', 'module', 'chip', 'sensor']
nrows = 100000 # load just the fist 100 000 of rows

df = read_root(path, tree, columns, nrows=nrows)

# Define CODEX-B acceptance geometry, model the acceptance as a cone of radius d at distance r from the interaction point

r = 10000 # radius of the cone in mm

x = 23725 # mm
y = 0 # mm
z = 12650 # mm

d = np.sqrt( x**2 + y**2 + z**2 )

theta_max = np.arctan(r/d)

codexb_center = np.array([x, y, z])
codexb_axis = codexb_center / np.linalg.norm(codexb_center)

# Compute angles

df['theta'] = compute_angles(codexb_axis, df)

# Check which hits are within the acceptance

df['in_codexb'] = df['theta'].abs() < theta_max

# Plot hits within acceptance

plt.figure()

plt.plot(df[df['in_codexb']]['z'], df[df['in_codexb']]['x'], '.', color = 'green', zorder=3, label='In CODEX-B acceptance')
plt.plot(df[~df['in_codexb']]['z'], df[~df['in_codexb']]['x'], '.', color = 'blue', zorder=2, label='Out of CODEX-B acceptance')

# plot the cone boundary (the factors are just for scaling purposes)

a_z = codexb_axis[2] * (np.cos(theta_max) - np.sin(theta_max)) * 70
a_x = codexb_axis[0] * (np.cos(theta_max) + np.sin(theta_max)) * 70

b_z = codexb_axis[2] * (np.cos(theta_max) + np.sin(theta_max)) * 150
b_x = codexb_axis[0] * (np.cos(theta_max) - np.sin(theta_max)) * 150

plt.plot([0, a_z], [0, a_x], 'r--', label='CODEX-B acceptance boundary', zorder=2)
plt.plot([0, b_z], [0, b_x], 'r--', zorder=4)


plt.grid(linestyle='--', zorder = 0)
plt.xlabel('z (mm)')
plt.ylabel('x (mm)')
plt.legend()

plt.savefig('first_plots/codexb_acceptance.png')
plt.close()

plt.figure()

plt.plot(df[df['in_codexb']]['x'], df[df['in_codexb']]['y'], '.', color = 'green', zorder=3, label='In CODEX-B acceptance')
plt.plot(df[~df['in_codexb']]['x'], df[~df['in_codexb']]['y'], '.', color = 'blue', zorder=2, label='Out of CODEX-B acceptance')

plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.legend()
plt.grid(linestyle='--')

plt.savefig('first_plots/codexb_acceptance_xy.png')

# Histograms

fig = plt.figure(figsize=(12,6))

ax = [fig.add_subplot(1,3, i+1) for i in range (3)]

df_in_codex = df[df['in_codexb']]

ls = ['module', 'chip', 'sensor']

for i, element in enumerate(ls):

    bins = np.arange(df_in_codex[element].min(), df_in_codex[element].max()+2) - 0.5

    ax[i].hist(df_in_codex[element], bins=bins, color='green', histtype='bar', alpha=0.7, label='In CODEX-B acceptance', edgecolor='black')
    ax[i].set_xlabel(element)
    ax[i].set_ylabel('Hits')
    ax[i].grid(linestyle=':')
    ax[i].legend()



fig.tight_layout()
plt.savefig('first_plots/histograms_module_chip_sensor_incodexacceptance.png')


# Now, we may calculate the solid angle subtended by each hit vs theta

Omega_0 = r**2 / d**2


df['solid_angle'] = Omega_0 * (d - df['x']*codexb_axis[0] - df['y']*codexb_axis[1] - df['z']*codexb_axis[2])

plt.figure()

plt.scatter(df['theta'], df['solid_angle'], s=1, alpha=0.5)
plt.xlabel('Theta (radians)')
plt.ylabel('Solid angle (sr)')
plt.grid(linestyle=':')

plt.savefig('first_plots/solid_angle_vs_theta.png')
plt.close()