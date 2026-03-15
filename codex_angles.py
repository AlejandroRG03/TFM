from functions import *

r = 10000 # radius of the cone in mm

x = 23725 # mm
y = 0 # mm
z = 12650 # mm

d = np.sqrt( x**2 + y**2 + z**2 )

codex_angular_aperture = np.arctan(r/d)

# check if the KL0 particles are within the acceptance

codexb_center = np.array([x, y, z])

theta_codexb = np.arccos(codexb_center[2] / np.linalg.norm(codexb_center))
phi_codexb   = np.arctan2(codexb_center[1], codexb_center[0])

min_theta_codex = theta_codexb - codex_angular_aperture
max_theta_codex = theta_codexb + codex_angular_aperture


print(f'CODEX-B angular acceptance: {codex_angular_aperture:.2f} radians')
print(f'CODEX-B axis theta: {theta_codexb:.2f} radians')
print(f'CODEX-B minimum theta: {min_theta_codex:.2f} radians')
print(f'CODEX-B maximum theta: {max_theta_codex:.2f} radians')