"""
    Normalizing the data (x,y) by baseline substraction. See Eq. (31) in [1].   

    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.
"""


import numpy as np

def preserve(x, y):
    return y - ((y[-1] - y[0]) / (x[-1] - x[0])) * x - ((x[-1] * y[0] - x[0] * y[-1]) / (x[-1] - x[0]))
