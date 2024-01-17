from mrs_ue_utils import generate_dem
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from skimage.restoration import unwrap_phase


def angle(v, w):
    return np.arccos(v.dot(w)/(norm(v)*norm(w)))

###########################
# Exercise 5
###########################


# Data import
path = "./data"

master = np.load(os.path.join(path, "master_img.npy"))
slave = np.load(os.path.join(path, "slave_img.npy"))


# Plot input data
fig, ax = plt.subplots(1,2, figsize=(20,20))
ax[0].imshow(master)
ax[1].imshow(slave)
ax[0].set_title('Master Image', fontsize=42)
ax[1].set_title('Slave Image', fontsize=42)
plt.show()

# Generate a flat earth DEM
x, y, z = generate_dem(z_min=0., z_max=0.)
dem = np.dstack((x, y, z))

# Position of Master and slave
p_m = np.array([0., -12_000., 10_000.])  # Master

# Direction and distance of slave
p_c = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.])
b = 100.

p_s = p_m - (p_c * b)  # Slave

baseline = (p_c * b)  # Baseline between Master and Slave

lambda_ = 0.05  # Wavelength

k = 2 * np.pi/lambda_  # Wave number


R1 = dem - p_m  # Distance between master and each point of the DEM
R2 = R1 - baseline  # Distance between slave and each point of the DEM

master_dem = norm(R1, axis=2) * k % (2 * np.pi)  # Phase at master
slave_dem = norm(R2, axis=2) * k % (2 * np.pi)  # Phase at slave

# compute angle alpha with respect to the local horizontal
loc_h = np.array([1, 0, 0])
alpha = angle(loc_h, p_c)

r1 = norm(R1, axis=2)
r2 = norm(R2, axis=2)

H = 10_000.
dem_xy = R1[:, :, :1]
theta_ref = np.arctan(y / H)
# # Compute look angle theta
# theta_l = np.arcsin(((r1**2)+(b**2)-(r2**2))/(2*r1*b))+alpha

gamma = angle(R1, baseline)
theta_l = (np.pi/2) - gamma + alpha
h = 10_000. - r1 * np.cos(theta_l)

#
# # Compute the flat earth contribution

d_phi = -((4 * np.pi / lambda_) * b * np.sin(theta_ref - alpha)) % (2 * np.pi)

interferogram = ((master-master_dem) - (slave-slave_dem)) % (2 * np.pi)
unwrapped_phases = unwrap_phase((interferogram) % (2*np.pi))



fig, ax = plt.subplots(1, 2, figsize=(20, 20))
ax[0].imshow(interferogram)
ax[1].imshow(unwrapped_phases)
ax[0].set_title('Interferogram', fontsize=42)
ax[1].set_title('Unwrapped phases', fontsize=42)
plt.show()