import numpy as np
from scipy.special import hermite as scipy_hermite

def hermite_system(alpha, k, b=1):
    """
    Compute the expansion matrix Phi (Nxk) that contains the values of the first k dilated Hermite functions.

    Parameters:
    - k     : values of the first 'k' Hermite functions will be calculated at 'alpha' 
    - alpha : values of the Hermite functions will be calculated at 'alpha'
              For instance, one can use the roots of the Nth Hermite polynomial.
    - b     : dilation parameter of the Hermite system (default is 1)

    Returns:
    - Phi : Phi[:,i] contains the values of the ith Hermite functions
    """
    Phi = 1 / np.sqrt(b * np.sqrt(np.pi)) * hermite(k, alpha, b)
    return Phi

def hermite(n, x, b):
    H = np.zeros((x.shape[0], n))
    x = np.reshape(x, (x.shape[0], 1))
    xx = np.reshape(x / b, (x.shape[0], 1))
    w = np.exp(-x**2 / (2 * b**2))  # weight function
    H[:, 0] = w[:, 0]               # zero order Hermite polynomial
    H[:, 1] = 2 * xx[:, 0] * w[:, 0] / np.sqrt(2)  # first order Hermite polynomial

    # Hermite polynomials by recursion
    for i in range(2, n):
        ni = 1 / np.sqrt(2 * i)
        ni_1 = 1 / np.sqrt(2 * (i - 1)) * ni
        H[:, i] = 2 * (xx[:, 0] * H[:, i - 1] * ni - (i - 1) * H[:, i - 2] * ni_1)
    
    return H
