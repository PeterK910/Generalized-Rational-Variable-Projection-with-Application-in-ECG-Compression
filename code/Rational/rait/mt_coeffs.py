from Rational.rait.mt_system import mt_system
import numpy as np


def mt_coeffs(v, poles):
    """
    MT_COEFFS - Coefficients of a vector in the Malmquist-Takenaka basis.

    Usage:
        [co,err] = mt_coeffs(v,poles)
    Input parameters:
        v     : an arbitrary vector
        poles : poles of the rational system
    Output parameters:
        co  : the Fourier coefficients of 'v' with respect to the MT system
            defined by poles
        err : L^2 norm of the approximation error
    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """

    np_ = poles.shape[0]
    nv, mv = v.shape
    if np_ != 1:
        raise ValueError("Wrong pole parameters!")
    if nv != 1:
        raise ValueError("Wrong vector to calc!")
    if np.max(np.abs(poles)) >= 1:
        raise ValueError("Bad poles!")

    mts = mt_system(mv, poles)
    co = (np.matmul(mts, v.T.conj()) / mv).T.conj()    
    err = np.linalg.norm(np.matmul(co, mts) - v)

    return co, err
