import numpy as np


def eBlaschke(pole, eps, z):
    Bz = (z - pole) / (1 - np.conj(pole) * z)
    Bz = eps * Bz
    return Bz
