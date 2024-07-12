from B_spline.solve_opt import solve_opt
import numpy as np
import pytest
import numpy.testing as npt


@pytest.mark.parametrize(
    "k,expected_c,expected_rll,expected_zll",
    [
        (
            5,
            [[0.0713], [0.2260], [0.2007]],
            [
                [27.313, 19.6976, 12.0822],
                [0, 23.0001, 14.0004],
                [32, 0, 20.0003],
                [46, 33, 0],
                [53, 38, 23],
            ],
            [[8.8236], [8.0085], [4.0134], [-2.6268]],
        ),
    ],
)
def test_1(k, expected_c, expected_rll, expected_zll):
    Rl = np.array([[1, 2], [3, 4], [4, 5], [6, 7], [7, 8]], dtype=np.float64)
    zl = np.array([[6], [7], [8], [4]], dtype=np.float64)
    B = np.array([[3, 2, 1], [4, 3, 2]], dtype=np.float64)
    b = np.array([[1, 2], [3, 4]])
    p = 2

    c, Rll, zll = solve_opt(Rl, zl, B, b, k, p)
    npt.assert_array_equal(np.round(c, 4), expected_c)
    npt.assert_array_equal(np.round(Rll, 4), expected_rll)
    npt.assert_array_equal(np.round(zll, 4), expected_zll)
