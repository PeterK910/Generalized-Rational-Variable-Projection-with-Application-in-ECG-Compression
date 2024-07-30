from Hermite.hermite_exp import quant
import numpy as np
import pytest

@pytest.mark.parametrize(
    "data,exp1,exp2",
    [
        ([0.010074,0.00026887], 0.007843, 0.0),
        ([-0.040827,0.052044], -0.039216, 0.054902),
    ],
)
def test_quant(data, exp1, exp2):    
    data=np.array(data)
    eps=8

    res=quant(data, eps)
    assert round(res[0], 6)==exp1
    assert round(res[1], 6)==exp2
