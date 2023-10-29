import numpy as np
from typing import Optional

VACUUM_PERMITTIVITY = 8.8541878128e-12 # F/m
VACUUM_PERMEABILITY = 1.25663706212e-6 # H/m
SPEED_OF_LIGHT = 299792458 # m/s

def emw(x: float, t: float, lambda_0: float, eps_prime: float, eps_double_prime: float, sigma: float, amplitude: Optional[float] = 1) -> float:
    '''Simple function representing an EMW, to calculate the electric field strength at a given position x at time t.
    Parameters:
    -----------
    x: float
        Position in m
    t: float
        Time in s
    lambda_0: float
        Wavelength in m
    eps_prime: float
        Real part of the relative permittivity
    eps_double_prime: float
        Imaginary part of the relative permittivity
    sigma: float
        Conductivity in S/m
    amplitude: float, optional
        Amplitude of the EMW, default is 1
    
    Returns:
    --------
    float
    '''
    
    omega = 2 * np.pi * SPEED_OF_LIGHT / lambda_0

    k_prime = omega * np.sqrt(eps_prime * VACUUM_PERMITTIVITY * VACUUM_PERMEABILITY / 2) * np.sqrt(np.sqrt(1 + (sigma / (eps_prime * VACUUM_PERMITTIVITY * omega))**2) + 1)
    try:    # try the case of a lossy medium
        k_double_prime = omega * np.sqrt(eps_double_prime * VACUUM_PERMITTIVITY * VACUUM_PERMEABILITY / 2) * np.sqrt(np.sqrt(1 + (sigma / (eps_double_prime * VACUUM_PERMITTIVITY * omega))**2) - 1)
        
        return (amplitude * np.exp(1j*(k_prime * x - omega * t))*np.exp(-k_double_prime * x)).real  # return the real part of the complex number
    
    except ZeroDivisionError:   # if sigma = 0, the medium is a dielectric or vacuum and the wavenumber is real
        k_0 = 2 * np.pi / lambda_0
        n = np.sqrt(eps_prime)
        k = k_0 * n
        return (amplitude * np.exp(1j*(k * x - omega * t))).real # return the real part of the complex number
    



position = np.linspace(0., 4*np.pi, 1000)
time = 0
wavelength = 0.19 # m, L1-band GPS with f = 1575.42 MHz
rel_permittivity_real = 1   # lets start in vaccum
rel_permittivity_imag = 0 # no losses, in vaccum
conductivity = 0 # no conductivity, in vaccum


efield = emw(x=position, t=time, lambda_0=wavelength, eps_prime=rel_permittivity_real, eps_double_prime=rel_permittivity_imag, sigma=conductivity)   