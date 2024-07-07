import numpy as np


def multiply_poles(p, m):
    """
    MULTIPLY_POLES - Duplicates the elements of 'p' by the elements of 'm'.

    Usage:
        pp = multiply_poles(p,m)

    Input parameters:
        p : row vector that contains a pole only once
        m : multiplicities related to the pole vector 'p'

    Output parameters:
        pp : vector of the poles that contains the ith element of 'p'
            only 'm(i)' times

    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """
    if len(p) != len(m):
        raise ValueError("Bad poles, length of p and m must be equal!")
    n = p.shape[0]
    pp = np.zeros(np.sum(m), dtype=p.dtype)
    innen = 0
    for i in range(n):
        pp[innen: innen + m[i]] = p[i] * np.ones(1, m[i])
        innen += m[i]

    return pp
