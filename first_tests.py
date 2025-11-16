from functions import * 

# Load data

path = '/scratch48/emilio.fernandez/Velo/VELOHits.root'
tree = 'VeloMultiTuple_73eaa531/Clusters;1'
columns = ['eventNumber', 'x', 'y', 'z']
nrows = 1_000_000 # load just the firt million of rows

df = read_root(path, tree, columns, nrows=nrows)

# Some plots

# transverse hits

plt.figure()

plt.plot(df['x'], df['y'], '.')
plt.grid(linestyle = '--')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('first_plots/transverse_hits.png')
plt.close()

# longitudinal hits

plt.figure()
plt.plot(df['z'], df['x'], '.')
plt.grid(linestyle='--')
plt.xlabel('z')
plt.ylabel('x')

plt.savefig('first_plots/longitudinal_hits.png')

# 3D plot

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')

# add rotation for better visualization
ax1.view_init(elev=5, azim=45)
ax2.view_init(elev=75, azim=0)

# sample 10000 points for better visualization

df_sampled = df.sample(n=10000, random_state=42)

ax1.scatter(df_sampled['x'], df_sampled['y'], df_sampled['z'], s=1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.grid(linestyle='--')

ax2.scatter(df_sampled['x'], df_sampled['y'], df_sampled['z'], s=1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.grid(linestyle='--')

fig.tight_layout()
plt.savefig('first_plots/3D_hits.png')
plt.close()