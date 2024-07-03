from Rational.rait.multiply_poles import multiply_poles
import numpy as np
import numpy.testing as npt

def test_1():
    p = np.array([0.5, 0.3, 0.1])  # Example poles
    m = np.array([2, 3, 1])  # Example multiplicities
    pp = multiply_poles(p, m)  # Compute duplicated poles
    print(pp)