"""
    Calculating the related Hermite coefficients 'co' of the signal 'sig'.
    Parameters:
    - signal  : samples of the signal (N=length(signal))
    - dilat   : dilation parameter of the Hermite system
    - trans   : translation parameter of the Hermite system
    - alpha   : roots of the Nth Hermite polyniomial
    - Lambda  : normalization parameters of the Hermite system (see, e.g., 'example.m')
    - HMS     : the ith row contains the values of the ith Hermite polynomial (i=0..N-1) at the roots 'alpha'

    Returns
    -  co : coefficients of the approximation

    Implemented by Peter Kovacs,
    Department of Numerical Analysis,
    E�tv�s Lorand University, Budapest, Hungary, 2015.

    This implementation is based on the following papers:

    [1] R. Jane, S. Olmos, P. Laguna, P. Caminal, 
        Adaptive Hermite models for ECG data compression: performance and evaluation with automatic wave detection, 
        Proceedings of Computers in Cardiology, 1993, pp. 389-392. 

    [2] A. Sandryhaila, S. Saba, M. Puschel, J. Kovacevic, 
        Efficient Compression of QRS Complexes Using Hermite Expansion, 
        IEEE Transactions on Signal Processing, vol. 60, no. 2, 2012, pp. 947-955. 

    [3] T. D�zsa, P. Kov�cs, 
        ECG signal compression using adaptive Hermite functions, 
        Advances in Intelligent Systems and Computing, vol. 399, 2015, pp. 245-254. 
"""

import numpy as np
from scipy.linalg import solve
from Hermite.hermite_roots import hermite_roots
from Hermite.subsample import subsample


def hermite_coeff(signal, dilat, trans, alpha=None, Lambda=None, HMS=None):
    N = len(signal)
    
    if alpha is None:
        # 'N' number of roots are used during the discreatization.
        alpha = hermite_roots(N)
    
    if N % 2 == 0:
        tk = np.arange(-N/2, N/2)
    else:
        tk = np.arange(-np.floor(N/2), np.floor(N/2) + 1)
    
    # Translating and dilating the original signal
    s = np.roll(signal, trans)  # translation
    s = subsample(s, dilat * tk, alpha)  # dilation
    
    # Computing the Hermite coefficients
    co = np.matmul(HMS.T, solve(Lambda, s.T))
    
    return co
