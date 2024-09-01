import torch
import torch.testing as tt
from Rational.with_torch.rait.util import addimag

def test_addimag():
    a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
    b = addimag(a)
    tt.assert_close(
        b, 
        torch.tensor([1 + 1.3764j, 2 - 0.52573j, 3 - 1.7013j, 4 - 0.52573j, 5 + 1.3764j], dtype=torch.cdouble),
        atol=1e-4,
        rtol=0
    )