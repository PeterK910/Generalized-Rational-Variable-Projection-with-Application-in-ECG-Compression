from Rational.rait.periodize_poles import periodize_poles
import numpy as np
import numpy.testing as npt


def test_1():
    p=np.array([1,2,3])
    pp=periodize_poles(p, 5)
    npt.assert_array_equal(pp, [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3])