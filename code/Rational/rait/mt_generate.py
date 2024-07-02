import numpy as np
from Rational.rait import mt_system

def mt_generate(len, poles, coeffs):
    """
    Generates a function in the space spanned by the MT system.

    Parameters:
        len : int
            Number of points in case of uniform sampling.
        poles : numpy array
            Poles of the rational system (row vector).
        coeffs : numpy array
            Coefficients of the linear combination to form (row vector).

    Returns:
        v : numpy array
            The generated function at the uniform sampling points as a row vector.

    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """
    np, mp = poles.shape
    nl, ml = coeffs.shape

    # Check input dimensions and conditions
    if np != 1 or nl != 1 or mp != ml or len < 2:
        raise ValueError('Wrong parameters!')
    if np.any(np.abs(poles) >= 1):
        raise ValueError('Bad poles!')

    # Calculate the linear combination of the MT system elements
    # using matrix multiplication
    v = np.matmul(coeffs, mt_system(len, poles))

    return v