from Rational.with_numpy.rait.addimag import addimag
import numpy as np
import numpy.testing as npt


def test_1():
    a = np.array([1, 2, 3, 4, 5])
    b = addimag(a)
    npt.assert_array_almost_equal(
        b, [1 + 1.3764j, 2 - 0.52573j, 3 - 1.7013j, 4 - 0.52573j, 5 + 1.3764j], 4
    )
