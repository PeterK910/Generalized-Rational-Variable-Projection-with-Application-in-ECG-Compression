import numpy as np
from scipy.interpolate import make_interp_spline

def subsample(sample, tk, xk):
    """
    SUBSAMPLE - Interpolates values between uniform sampling points.

    Usage:
        y = subsample(sample, tk, xk)

    Input parameters:
        sample : row vector of uniformly sampled values on tk 
        tk     : the values are given at tk
        xk     : the values at xk is to be found

    Output parameters:
        y : the interpolated value at the point(s) xk according to sample
    """

    f=make_interp_spline(tk, sample)
    return f(xk)
    