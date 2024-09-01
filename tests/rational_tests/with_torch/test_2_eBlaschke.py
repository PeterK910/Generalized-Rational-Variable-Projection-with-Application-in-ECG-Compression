import torch
import torch.testing as tt
from Rational.with_torch.Hyperbolic_operators.eBlaschke import eBlaschke


def test_eBlaschke():
    pole = torch.tensor(0.5 + 0.3j, dtype=torch.complex128)
    eps = torch.tensor(1.2, dtype=torch.float64)
    z = torch.exp(1j * torch.linspace(-torch.pi, torch.pi, 4, dtype=torch.complex128))
    expected = torch.tensor([-1.1077 - 0.46154j, -0.6 - 1.0392j, -0.6 + 1.0392j, -1.1077 - 0.46154j], dtype=torch.complex128)
    tt.assert_close(eBlaschke(pole, eps, z), expected, atol=1e-4, rtol=0)
