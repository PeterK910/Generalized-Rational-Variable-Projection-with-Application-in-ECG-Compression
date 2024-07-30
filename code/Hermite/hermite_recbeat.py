import numpy as np
from Hermite.hermite_system import hermite_system


def hermite_recbeat(bm, la, co, b, tk=None):
    """
    Reconstruct the heartbeat using Hermite systems with various dilation parameters.
    Each dilated Hermite system corresponds to one of the main waves (P, QRS, T) of the heart beat.
    Note: it is possible to use more than three systems for the reconstruction, e.g., for the ST segment.
    
    Parameters:
        - bm  : The heart beat is segmented into different sections (or waves):
            - 'bm[i, 0]' contains the length of the ith section
            - 'bm[i, 1]' contains the translation parameter of the Hermite system of ith section
        - la  : contains the boundary values of each section
        - co  : it is a list, co[i] contains the coefficients of the ith section 
        - b   : 'b[i]' contains the dilation parameter of the Hermite system of ith section
        - tk  : we calculate the values of the Hermite representation or approximation at 'tk'

    Returns:
        - sig : reconstructed heart beat (approximation)

    Implemented by Peter Kovacs,
    Department of Numerical Analysis,
    E�tv�s Lorand University, Budapest, Hungary, 2015.

    This implementation is based on the following papers:

    [1] R. Jane, S. Olmos, P. Laguna, P. Caminal, 
        Adaptive Hermite models for ECG data compression: performance and evaluation with automatic wave detection, 
        Proceedings of Computers in Cardiology, 1993, pp. 389-392. 

    [2] T. D�zsa, P. Kov�cs, 
        ECG signal compression using adaptive Hermite functions, 
        Advances in Intelligent Systems and Computing, vol. 399, 2015, pp. 245-254. 

    """
    if tk is None:
        tk = []
        for i in range(len(b)):
            M = bm[i, 0]
            if M % 2 == 0:
                tk.append(np.arange(-M//2, M//2))
            else:
                tk.append(np.arange(-M//2, M//2 + 1))

    # Signal reconstruction
    aprx = []
    for i in range(len(co)):
        n = co[i].toarray().shape[0]
        x = np.arange(1, bm[i, 0] + 1) - bm[i, 1]
        uniform_hms = hermite_system(tk[i].toarray() / b[i], n)
        aprx.append((np.matmul(uniform_hms, co[i].toarray())).T)

    # Baseline or trend reconstruction
    sig = []
    for i in range(len(la) - 1):
        m = (la[i + 1] - la[i]) / (len(aprx[i][0]) - 1)
        x = np.arange(len(aprx[i][0]))
        base_line = la[i] + m * x
        aprx[i][0] = aprx[i][0] + base_line
        sig.extend(aprx[i][0][:-1])
    sig.extend(aprx[-1][0][-1:])

    return np.array(sig)

