from Rational.Hyperbolic_operators.rho0 import rho0
import numpy as np
import numpy.testing as npt

def test_1():
    z1=np.array([-1,4,8])
    z2=np.array([6,3,9])
    npt.assert_array_almost_equal(rho0(z1, z2), [1, 0.090909, 0.014085])