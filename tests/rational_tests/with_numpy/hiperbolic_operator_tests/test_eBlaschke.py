from Rational.Hyperbolic_operators.eBlaschke import eBlaschke
import numpy as np
import numpy.testing as npt


def test_1():
    pole = 0.5 + 0.3j
    eps = 1.2
    z = np.exp(1j * np.linspace(-np.pi, np.pi, 4))
    Bz = eBlaschke(pole, eps, z)
    npt.assert_array_almost_equal(
        Bz, [-1.1077 - 0.46154j, -0.6 - 1.0392j, -0.6 + 1.0392j, -1.1077 - 0.46154j], 4
    )
