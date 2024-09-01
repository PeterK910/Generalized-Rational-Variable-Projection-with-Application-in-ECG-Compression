from Rational.with_numpy.Hyperbolic_operators.rho import rho
import numpy as np
import numpy.testing as npt


def test_1():
    z1 = np.array([2, 4, 8])
    z2 = np.array([6, 3, 9])
    npt.assert_array_almost_equal(rho(z1, z2), [0.38107, 0.091161, 0.014085])
