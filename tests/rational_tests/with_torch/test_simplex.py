import torch
import pytest
import torch.testing as tt
from Rational.with_torch.rait.simplex import periodize_poles, multiply_poles


def test_periodize_poles():
    p = torch.tensor([1, 2, 3], dtype=torch.float64)
    pp = periodize_poles(p, 5)
    tt.assert_close(pp, torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=torch.float64))

@pytest.mark.parametrize(
    "in_p,in_m,out",
    [
        ([0.12+0.34j], [1], [0.12+0.34j]),
        ([0.12+0.34j, 0.2-0.3j, 0.3+0.4j], [2, 3, 1], [0.12+0.34j, 0.12+0.34j, 0.2-0.3j, 0.2-0.3j, 0.2-0.3j, 0.3+0.4j]),
    ],
)
def test_multiply_poles(in_p, in_m, out):
    p = torch.tensor(in_p, dtype=torch.complex64)
    m = torch.tensor(in_m, dtype=torch.int64)
    pp = multiply_poles(p, m)
    print(pp)
    tt.assert_close(pp, torch.tensor(out, dtype=torch.complex64))
