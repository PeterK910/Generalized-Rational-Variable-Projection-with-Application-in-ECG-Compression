from Rational import norm_sig as testing
import numpy as np
import pytest
import numpy.testing as npt

@pytest.mark.parametrize("input,expected_norm,expected_base", [
    ([1,2,3,4,5], [0,0,0,0,0], [1,2,3,4,5]),
    ([1,5,3,4,5], [0,3,0,0,0], [1,2,3,4,5]),
    ([1,5,4,3,5], [0,3,1,-1,0], [1,2,3,4,5]),
])
def test_1(input, expected_norm, expected_base):
    signal = np.array(input)
    norm, base=testing.norm_sig(signal)
    npt.assert_array_equal(norm, expected_norm)
    npt.assert_array_equal(base, expected_base)
