import torch
import torch.testing as tt
from Rational.with_torch.Hyperbolic_operators.rho0 import rho0
from Rational.with_torch.Hyperbolic_operators.rho import rho


def test_rho_0():
    z1 = torch.tensor([-1, 4, 8], dtype=torch.float64)
    z2 = torch.tensor([6, 3, 9], dtype=torch.float64)
    expected = torch.tensor([1, 0.090909, 0.014085], dtype=torch.float64)
    tt.assert_close(rho0(z1, z2), expected, atol=1e-4, rtol=0)

def test_rho():
    z1 = torch.tensor([2, 4, 8], dtype=torch.float64)
    z2 = torch.tensor([6, 3, 9], dtype=torch.float64)
    expected = torch.tensor([0.38107, 0.091161, 0.014085], dtype=torch.float64)
    tt.assert_close(rho(z1, z2), expected, atol=1e-4, rtol=0)