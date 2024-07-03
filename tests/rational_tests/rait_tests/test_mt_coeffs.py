from Rational.rait.mt_coeffs import mt_coeffs
import numpy as np
import numpy.testing as npt

def test_1():    
    v = np.array([[1,2,3]])
    poles=np.array([[0.5, 0.3, 0.1]])
    co, err = mt_coeffs(v, poles)
    npt.assert_array_almost_equal(co, [[1.6358-0.16667j,-1.5527-0.24401j,-1.0456+0.21975j]], 4)
    npt.assert_almost_equal(err, np.float64(1.5321), 4)