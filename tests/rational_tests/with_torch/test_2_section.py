import torch
import torch.testing as tt
from Rational.with_torch.Hyperbolic_operators.section import section


def test_section():
    w1 = torch.tensor(0.5 + 0.5j, dtype=torch.complex64)
    w2 = torch.tensor(-0.5 - 0.5j, dtype=torch.complex64)
    w, pole, eps, p = section(w1, w2, resolution=8, draw=False)
    expected_w = torch.tensor(
        [
            0.5 + 0.5j,
            0.44737 + 0.44737j,
            0.38235 + 0.38235j,
            0.3 + 0.3j,
            0.19231 + 0.19231j,
            0.045455 + 0.045455j,
            -0.16667 - 0.16667j,
            -0.5 - 0.5j,
        ], dtype=torch.complex64
    )
    tt.assert_close(w, expected_w, atol=1e-5, rtol=0)
    tt.assert_close(pole, torch.tensor(7.071068e-01, dtype=torch.complex64), atol=1e-6, rtol=0)
    tt.assert_close(eps, torch.tensor(-7.071068e-01 - 7.071068e-01j, dtype=torch.complex64), atol=1e-6, rtol=0)
    tt.assert_close(p, torch.tensor(9.428090e-01, dtype=torch.float32), atol=1e-6, rtol=0)
