from Rational.with_numpy.rait.multiply_poles import multiply_poles
import numpy as np
import pytest
import numpy.testing as npt

@pytest.mark.parametrize(
    "in_p,in_m,out",
    [
        ([1], [1], [1]),
        ([1,2,3], [2, 3, 1], [1,1,2,2,2,3]),
    ],
)
def test_1(in_p, in_m, out):
    p = np.array(in_p)
    m = np.array(in_m)
    pp = multiply_poles(p, m)
    npt.assert_array_equal(pp, out)
