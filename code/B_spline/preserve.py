"""
    Normalizing the data (x,y) by baseline substraction. See Eq. (31) in [1].   

    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.
"""


import numpy as np

def preserve(x, y):
    return y - ((y[-1] - y[0]) / (x[-1] - x[0])) * x - ((x[-1] * y[0] - x[0] * y[-1]) / (x[-1] - x[0]))

# Example usage:
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

yy = preserve(x, y)
print("yy:\n", yy)
