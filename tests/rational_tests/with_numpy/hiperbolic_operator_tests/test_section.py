from Rational.Hyperbolic_operators.section import section
import numpy as np
import numpy.testing as npt


def test_1():
    w1 = 0.5 + 0.5j
    w2 = -0.5 - 0.5j
    w, pole, eps, p = section(w1, w2, resolution=8, draw=False)
    npt.assert_array_almost_equal(
        w,
        [
            0.5 + 0.5j,
            0.44737 + 0.44737j,
            0.38235 + 0.38235j,
            0.3 + 0.3j,
            0.19231 + 0.19231j,
            0.045455 + 0.045455j,
            -0.16667 - 0.16667j,
            -0.5 - 0.5j,
        ],
        5,
    )
    npt.assert_almost_equal(pole, np.float64(7.071068e-01), 6)
    npt.assert_almost_equal(eps, np.complex64(-7.071068e-01 - 7.071068e-01j), 6)
    npt.assert_almost_equal(p, np.float64(9.428090e-01), 6)
