from B_spline.Bspline import Bspline
import numpy as np
import pytest
import numpy.testing as npt

@pytest.mark.parametrize(
    "l,k,xk,expected",
    [
        (0, 2, [0, 1, 2, 3, 4, 5, 6], [0,0,1,0,0,0,0,0,0,0]),
        (1, 2, [0, 1, 2, 3, 4, 5, 6], [0,0,0.33333,1,0.33333,0,0,0,0,0]),
    ],
)
def test_1(l, k, xk, expected):
    x = np.linspace(0, 6, 10)
    y = Bspline(l, k, xk, x)
    npt.assert_array_almost_equal(y, expected, 5)