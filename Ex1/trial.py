import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def func(k, n, w, theta):
    return (np.exp(1j * k * n * w (np.sin(theta) + 1)) / (k * n * (np.sin(theta) + 1)))

thetas = np.linspace(-np.pi / 2, np.pi / 2, 1000)
k = 2 * np.pi / 0.19
n = 1
w = 0.1 

pattern = func(k, n, w, thetas)

fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', which='both', direction = 'in', top=True, bottom=True, left=True, right=True)
ax.plot(thetas, func(k, n, w, thetas))
