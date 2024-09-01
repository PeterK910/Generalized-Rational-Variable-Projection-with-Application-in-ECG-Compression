from Rational.with_numpy.Hyperbolic_operators.scale import scale
import numpy.testing as npt


def test_1():
    s, w = scale(2, 3, 0.5, False)
    npt.assert_almost_equal(s, 0.10102, 4)
    npt.assert_almost_equal(w, 2.3798, 4)
