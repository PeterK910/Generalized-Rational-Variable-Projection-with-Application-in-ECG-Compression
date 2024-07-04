import numpy as np


def mt_system(len, poles):
    """
    Generates the Malmquist-Takenaka system.

    Parameters:
        len : int
            Number of points in case of uniform sampling.
        poles : numpy array
            Poles of the rational system.

    Returns:
        mts : numpy array
            The elements of the MT system at the uniform sampling points as row vectors.

    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """
    np_, mp = poles.shape
    if np_ != 1 or len < 2:
        raise ValueError("Wrong parameters!")
    if np.any(np.abs(poles) >= 1):
        raise ValueError("Bad poles!")

    mts = np.zeros((mp, len), dtype=complex)
    t = np.linspace(-np.pi, np.pi, len + 1)[:-1]
    z = np.exp(1j * t)

    fi = np.ones(len, dtype=complex)  # the product defining MT elements so far
    for j in range(mp):
        co = np.sqrt(1 - (np.abs(poles[0, j]) ** 2))
        rec = 1 / (1 - np.conj(poles[0, j]) * z)
        lin = co * rec
        bla = (z - poles[0, j]) * rec
        mts[j, :] = lin * fi
        fi = fi * bla

    return mts
