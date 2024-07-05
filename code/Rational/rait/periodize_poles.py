import numpy


def periodize_poles(p, m):
    """
    PERIODIZE_POLES - Duplicates periodically the elements of 'p' 'm' times.

    Usage:
        pp=periodize_poles(p,m)

    Input parameters:
        p : row vector that contains the poles
        m : integer factor of duplication

    Output parameters:
        pp : vector of the poles that contains 'p' sequentially

    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """

    pp = numpy.zeros(m * len(p), dtype=complex)
    for i in range(1, m + 1):
        pp[(i - 1) * len(p): i * len(p)] = p

    return pp
