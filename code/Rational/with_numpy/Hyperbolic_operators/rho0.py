import numpy as np


def rho0(z1, z2):
    return np.abs(z1 - z2) / np.abs(1 - np.conj(z2) * z1)
