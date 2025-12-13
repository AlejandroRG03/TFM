from functions import *

# Load data

path = '/scratch48/emilio.fernandez/Velo/VELOHits.root'
tree = 'VeloMultiTuple_73eaa531/Clusters;1'
columns = ['eventNumber', 'x', 'y', 'z', 'module', 'chip', 'sensor']
nrows = 100000 # load just the fist 100 000 of rows

df = read_root(path, tree, columns, nrows=nrows)

# Define spherical variables

df['r'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['theta'] = np.arctan2(np.sqrt(df['x']**2 + df['y']**2), df['z'])
df['phi'] = np.arctan2(df['y'], df['x'])

df['eta'] = -np.log(np.tan(df['theta'] / 2))

# Define CODEX-B acceptance geometry, model the acceptance as a cone of radius d at distance r from the interaction point

r = 10000 # radius of the cone in mm

x = 23725 # mm
y = 0 # mm
z = 12650 # mm

d = np.sqrt( x**2 + y**2 + z**2 )

codex_angular_aperture = np.arctan(r/d)

print(f'Codex angular aperture = {codex_angular_aperture * 180 / np.pi : .2f} degrees')

codexb_center = np.array([x, y, z])
codexb_axis = codexb_center / np.linalg.norm(codexb_center)

# Compute angles

df['codex_angles'] = compute_angles(codexb_axis, df)

# Check which hits are within the acceptance

df['in_codex'] = df['codex_angles'] < codex_angular_aperture

df_in_codex = df[df['in_codex']]
df_not_in_codex = df[~df['in_codex']]

# Plot hits within acceptance

plt.figure()

plt.plot(df_in_codex['z'], df_in_codex['x'], '.', color = 'green', zorder=3, label='In CODEX acceptance')
plt.plot(df_not_in_codex['z'], df_not_in_codex['x'], '.', color = 'blue', zorder=2, label='Out of CODEX acceptance')

# plot the cone boundary (the factors are just for scaling purposes)

a_z = codexb_axis[2] * (np.cos(codex_angular_aperture) - np.sin(codex_angular_aperture)) * 70
a_x = codexb_axis[0] * (np.cos(codex_angular_aperture) + np.sin(codex_angular_aperture)) * 70

b_z = codexb_axis[2] * (np.cos(codex_angular_aperture) + np.sin(codex_angular_aperture)) * 150
b_x = codexb_axis[0] * (np.cos(codex_angular_aperture) - np.sin(codex_angular_aperture)) * 150

plt.plot([0, a_z], [0, a_x], 'r--', label='CODEX-B acceptance boundary', zorder=2)
plt.plot([0, b_z], [0, b_x], 'r--', zorder=4)


plt.grid(linestyle='--', zorder = 0)
plt.xlabel('z (mm)')
plt.ylabel('x (mm)')
plt.legend()

plt.savefig('first_plots/codexb_acceptance.png')
plt.close()

plt.figure()

plt.plot(df_in_codex['x'], df_in_codex['y'], '.', color = 'green', zorder=3, label='In CODEX acceptance')
plt.plot(df_not_in_codex['x'], df_not_in_codex['y'], '.', color = 'blue', zorder=2, label='Out of CODEX acceptance')

plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.legend()
plt.grid(linestyle='--')

plt.savefig('first_plots/codexb_acceptance_xy.png')

# Histograms

fig = plt.figure(figsize=(12,6))

ax = [fig.add_subplot(1,3, i+1) for i in range (3)]


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

CODEX_area = np.pi * r**2

v_x = codexb_center[0] - df['x'].values
v_y = codexb_center[1] - df['y'].values
v_z = codexb_center[2] - df['z'].values

dist = np.sqrt(v_x**2 + v_y**2 + v_z**2)

df['distance_to_codex'] = dist

cosines = (codexb_axis[0] * v_x + codexb_axis[1] * v_y + codexb_axis[2] * v_z) / dist

df['solid_angle'] = CODEX_area / dist**2 * cosines


df_in_codex = df[df['in_codex']]
df_not_in_codex = df[~df['in_codex']]

# Solid angle vs theta

plt.figure()

plt.scatter(df_in_codex['theta'] * 180/np.pi, df_in_codex['solid_angle'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['theta'] * 180/np.pi, df_not_in_codex['solid_angle'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('$\\theta$ (degrees)')
plt.ylabel('$\\Omega$ (sr)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/solid_angle_vs_theta.png')
plt.close()

# Solid angle vs eta

plt.figure()

plt.scatter(df_in_codex['eta'], df_in_codex['solid_angle'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['eta'], df_not_in_codex['solid_angle'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('$\\eta$')
plt.ylabel('$\\Omega$ (sr)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/solid_angle_vs_eta.png')
plt.close()

# Solid angle vs distance to codex

plt.figure()

plt.scatter(df_in_codex['distance_to_codex'], df_in_codex['solid_angle'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['distance_to_codex'], df_not_in_codex['solid_angle'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('Distance to codex (m)')
plt.ylabel('$\\Omega$ (sr)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/solid_angle_vs_distance.png')
plt.close()

# Solid angle vs codex angles

plt.figure()

plt.scatter(df_in_codex['codex_angles'] * 180 / np.pi, df_in_codex['solid_angle'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['codex_angles'] * 180 / np.pi, df_not_in_codex['solid_angle'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('$\\psi$ (degrees)')
plt.ylabel('$\\Omega$ (sr)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/solid_angle_vs_codex_angle.png')
plt.close()

# Solid angle vs distance to PV

plt.figure()

plt.scatter(df_in_codex['r'], df_in_codex['solid_angle'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['r'], df_not_in_codex['solid_angle'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('Distance to PV (mm)')
plt.ylabel('$\\Omega$ (sr)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/solid_angle_vs_distance_to_PV.png')
plt.close()

# distance to PV vs distance to codex

plt.figure()

plt.scatter(df_in_codex['r'], df_in_codex['distance_to_codex'], s = 1, color = 'green', alpha = 0.6, label='In CODEX acceptance')
plt.scatter(df_not_in_codex['r'], df_not_in_codex['distance_to_codex'], s = 1, alpha = 0.6, color = 'blue', label='Out of CODEX acceptance')

plt.xlabel('Distance to PV (mm)')
plt.ylabel('Distance to codex (mm)')
plt.grid(linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('first_plots/distance_to_PV_vs_distance_to_codex.png')
plt.close()


# Histograms of codex variables


variables = ['r', 'distance_to_codex', 'solid_angle', 'theta', 'eta', 'codex_angles']

for variable in variables:

    plt.figure()

    plt.hist(df_in_codex[variable], bins=50, color='green', alpha=0.6, label='In CODEX acceptance')

    plt.xlabel(variable)
    plt.ylabel('Counts')
    plt.grid(linestyle=':')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'first_plots/histogram_{variable}.png')
    plt.close()

