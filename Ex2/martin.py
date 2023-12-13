import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Label
import numpy as np

import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning,)

EPS_INF_WATER_20 = 4.9
EPS_S_WATER_20 = 80
NU_0_WATER_20 = 17e12

def debye_function(eps_inf: float, eps_s: float, nu_0: float, nu: float) -> complex:
    """
    Return the Debye function for a given set of parameters.

    Parameters
    ----------

    eps_inf : float
        High frequency dielectric constant

    eps_s : float
        Static dielectric constant

    nu_0 : float
        Characteristic frequency (Hz)

    nu : float
        Frequency (Hz)

    Returns
    -------

    eps : complex
        Dielectric constant
    """


    return eps_inf + ((eps_s - eps_inf) / (1 + (1j * nu / nu_0)))


freqs = np.logspace(1, 20, 10000)

# with plt.style.context('science'):
fig = plt.figure(figsize=(14, 6))
fig.suptitle('Debye function of water at 20°C', fontsize=16)

ax = fig.add_subplot(211)
ax.plot(freqs, np.real(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'black', linewidth = 2, label=r'$\epsilon_r\prime$')
ax.plot(freqs, -np.imag(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'red', linewidth = 2, label=r'$\epsilon_r\prime\prime$')
ax.set_xlabel(r'Frequency $\nu$ (Hz) ')
ax.set_ylabel(r'Debye Function $\epsilon_r$')
ax.legend()

ax.set_xscale('log')

ax0 = fig.add_subplot(223)
ax0.set_title('VIS Regime')
freqs = np.linspace(400e12, 800e12, 1000)
ax0.plot(freqs, np.real(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'black', linewidth = 2, label=r'$\epsilon_r\prime$')
ax0.plot(freqs, -np.imag(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'red', linewidth = 2, label=r'$\epsilon_r\prime\prime$')
ax0.set_xlabel(r'Frequency $\nu$ (Hz) ')
ax0.set_ylabel(r'Debye Function $\epsilon_r$')
ax0.legend()
ax0.set_xscale('log')

ax1 = fig.add_subplot(224)
freqs = np.linspace(0.3e12, 300e12, 1000)
ax1.set_title('MW Regime')
ax1.plot(freqs, np.real(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'black', linewidth = 2, label=r'$\epsilon_r\prime$')
ax1.plot(freqs, -np.imag(debye_function(eps_inf=EPS_INF_WATER_20, eps_s=EPS_S_WATER_20, nu_0=NU_0_WATER_20, nu=freqs)), color = 'red', linewidth = 2, label=r'$\epsilon_r\prime\prime$')
ax1.set_xlabel(r'Frequency $\nu$ (Hz) ')
ax1.set_ylabel(r'Debye Function $\epsilon_r$')
ax1.legend()
ax1.set_xscale('log')
plt.tight_layout()
plt.show()