from Rational.with_torch.Hyperbolic_operators.section import section
from Rational.with_torch.Hyperbolic_operators.eBlaschke import eBlaschke
import matplotlib.pyplot as plt
import torch

def custom_atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

def scale(w1, w2, beta, draw, color=[0, 0, 0]):
    if w1 == w2:
        return 0, 0
    else:
        w, pole, eps, p = section(w1, w2, 100, draw, color)
        s05 = torch.tanh(beta * custom_atanh(p))
        w05 = eBlaschke(pole, eps, s05)
        if draw and w.is_complex():
            if isinstance(color, str):
                plt.plot(w05.real.numpy(), w05.imag.numpy(), color, markersize=10)
            else:
                plt.plot(w05.real.numpy(), w05.imag.numpy(), color=".", markersize=10)
        return s05, w05
