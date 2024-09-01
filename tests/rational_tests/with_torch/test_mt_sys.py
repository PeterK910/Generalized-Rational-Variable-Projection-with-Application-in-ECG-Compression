import torch
import torch.testing as tt
from Rational.with_torch.rait.mt_sys import mt_system, mt_generate, mt_coeffs


def test_mt_system():
    poles = torch.tensor([0.5, 0.3, 0.1], dtype=torch.complex64)
    mts = mt_system(5, poles)
    expected_mts = torch.tensor([
        [
            0.5774 - 0.0000j,
            0.6413 - 0.2642j,
            1.1695 - 0.5772j,
            1.1695 + 0.5772j,
            0.6413 + 0.2642j,
        ],
        [
            -0.7338 - 0.0000j,
            -0.8244 - 0.1842j,
            -0.2476 - 1.2016j,
            -0.2476 + 1.2016j,
            -0.8244 + 0.1842j,
        ],
        [
            0.9045 + 0.0000j,
            0.4109 + 0.8688j,
            -0.9744 - 0.4666j,
            -0.9744 + 0.4666j,
            0.4109 - 0.8688j,
        ],
    ], dtype=torch.complex64)

    tt.assert_close(mts, expected_mts, atol=1e-4, rtol=0)


def test_mt_generate():
    poles = torch.tensor([0.5, 0.3, 0.1], dtype=torch.cdouble)
    coeffs = torch.tensor([2, 4, 3], dtype=torch.complex64)
    v = mt_generate(4, poles, coeffs)
    tt.assert_close(
        v, 
        torch.tensor([0.933105+1.755e-16j, -1.92899+1.01481j, 12.2318+0j, -1.92899-1.01481j], dtype=torch.complex64),
        atol=1e-5,
        rtol=0
    )

def test_mt_coeffs():    
    v = torch.tensor([1, 2, 3], dtype=torch.cfloat)
    poles = torch.tensor([0.5, 0.3, 0.1], dtype=torch.cfloat)
    co, err = mt_coeffs(v, poles)
    tt.assert_close(
        co, 
        torch.tensor([1.6358-0.16667j, -1.5527-0.24401j, -1.0456+0.21975j], dtype=torch.complex64),
        atol=1e-4,
        rtol=0
    )
    tt.assert_close(err, 1.5321, atol=1e-4, rtol=0)
