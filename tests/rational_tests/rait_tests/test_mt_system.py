from Rational.rait import mt_system as testing
import numpy as np
import numpy.testing as npt

def test_1():
    poles = np.array([[0.5, 0.3, 0.1]])  
    mts = testing.mt_system(5, poles)
    npt.assert_array_almost_equal(mts, [[0.5774 - 0.0000j, 0.6413 - 0.2642j, 1.1695 - 0.5772j, 1.1695 + 0.5772j, 0.6413 + 0.2642j ],
                                        [-0.7338 - 0.0000j, -0.8244 - 0.1842j, -0.2476 - 1.2016j, -0.2476 + 1.2016j, -0.8244 + 0.1842j ],
                                        [0.9045 + 0.0000j, 0.4109 + 0.8688j, -0.9744 - 0.4666j, -0.9744 + 0.4666j, .4109 - 0.8688j ]], 4)