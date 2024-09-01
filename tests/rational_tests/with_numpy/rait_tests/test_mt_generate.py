from Rational.rait.mt_generate import mt_generate
import numpy as np
import numpy.testing as npt

def test_1():
    poles = np.array([[0.5, 0.3, 0.1]])
    coeffs=np.array([[2,4,3]])
    v=mt_generate(4, poles, coeffs)
    npt.assert_array_almost_equal(v, [[0.933105+1.755e-16j, -1.92899+1.01481j, 12.2318+0j, -1.92899-1.01481j]], 5)