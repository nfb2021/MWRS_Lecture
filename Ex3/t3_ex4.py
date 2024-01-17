from mrs_ue_utils import generate_dem
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from skimage.restoration import unwrap_phase


def angle(v, w):
    return np.arccos(v.dot(w)/(norm(v)*norm(w)))


x, y, z = generate_dem()
dem = np.dstack((x, y, z))

# Position master
p_m = np.array([0., -12_000., 10_000.])
p_c = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.])
b = 100.

p_m = p_m.astype(np.longdouble)
p_c = p_c.astype(np.longdouble)
dem = dem.astype(np.longdouble)

p_s = p_m - (p_c * b)

baseline = (p_c * b)

lambda_ = 0.05

k = 2 * np.pi/lambda_


R1 = dem - p_m
R2 = R1 - baseline

# Show original image
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(z)
ax.set_title('Input DEM', fontsize=42)
plt.show()

master = norm(R1, axis=2) * k % (2 * np.pi)
slave = norm(R2, axis=2) * k % (2 * np.pi)
interferogram = (master-slave) % (2 * np.pi)

unwrapped_phases = unwrap_phase(interferogram)

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(master)
ax[1].imshow(slave)
ax[0].set_title('Master Image', fontsize=42)
ax[1].set_title('Slave Image', fontsize=42)

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(interferogram)
ax[1].imshow(unwrapped_phases)

ax[0].set_title('Interferogram', fontsize=42)
ax[1].set_title('Interferogram unwrapped', fontsize=42)
plt.show()


