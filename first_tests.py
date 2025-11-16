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