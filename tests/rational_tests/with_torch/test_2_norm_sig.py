import torch
import pytest
import torch.testing as tt
from Rational.with_torch.norm_sig import norm_sig


@pytest.mark.parametrize(
    "input,expected_norm,expected_base",
    [
        ([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5]),
        ([1, 5, 3, 4, 5], [0, 3, 0, 0, 0], [1, 2, 3, 4, 5]),
        ([1, 5, 4, 3, 5], [0, 3, 1, -1, 0], [1, 2, 3, 4, 5]),
    ],
)
def test_norm_sig(input, expected_norm, expected_base):
    signal = torch.tensor(input, dtype=torch.float64)
    norm, base = norm_sig(signal)
    tt.assert_close(norm, torch.tensor(expected_norm, dtype=torch.float64), atol=1e-4, rtol=0)
    tt.assert_close(base, torch.tensor(expected_base, dtype=torch.float64), atol=1e-4, rtol=0)
