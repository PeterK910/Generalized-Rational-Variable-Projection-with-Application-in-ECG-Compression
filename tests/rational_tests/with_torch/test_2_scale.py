import torch
import torch.testing as tt
from Rational.with_torch.Hyperbolic_operators.scale import scale


def test_scale():
    s, w = scale(torch.tensor(2), 3, 0.5, False)
    tt.assert_close(s, torch.tensor(0.10102, dtype=torch.float32), atol=1e-4, rtol=0)
    tt.assert_close(w, torch.tensor(2.3798, dtype=torch.float32), atol=1e-4, rtol=0)
