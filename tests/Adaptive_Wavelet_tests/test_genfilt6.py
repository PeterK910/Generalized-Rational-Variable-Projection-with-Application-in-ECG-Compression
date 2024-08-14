from Adaptive_Wavelets.genfilt6 import genfilt6
import numpy as np
import numpy.testing as npt

def test_1():
    theta1=1.3605
    theta2=-0.7916
    lo_d, hi_d, lo_r, hi_r = genfilt6(theta1, theta2)
    
    wanted_lor=np.array([ 0.3374, 0.8076, 0.4549, -0.1361, -0.0852, 0.0356])
    wanted_hir=np.array([0.0356, 0.0852, -0.1361, -0.4549, 0.8076, -0.3374])
    wanted_lod=np.array([ 0.0356, -0.0852, -0.1361, 0.4549, 0.8076, 0.3374])
    wanted_hid=np.array([-0.3374, 0.8076, -0.4549, -0.1361, 0.0852, 0.0356])

    npt.assert_array_almost_equal(lo_r, wanted_lor, 4)
    npt.assert_array_almost_equal(hi_r, wanted_hir, 4)
    npt.assert_array_almost_equal(lo_d, wanted_lod, 4)
    npt.assert_array_almost_equal(hi_d, wanted_hid, 4)
