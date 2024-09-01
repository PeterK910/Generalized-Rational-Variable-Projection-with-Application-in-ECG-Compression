from Rational.with_torch.Hyperbolic_operators.eBlaschke import eBlaschke
import matplotlib.pyplot as plt
import torch

def section(w1, w2, resolution, draw, color=[0, 0, 0], style="-"):
    W = eBlaschke(w1, 1, w2)
    eps = W / torch.abs(W)
    pole = -torch.conj(eps) * w1
    p = torch.abs(W)
    t = torch.linspace(0, p.squeeze(), resolution)
    w = eBlaschke(pole, eps, t)

    if draw and w.is_complex():
        if isinstance(color, str):
            plt.plot(w.real.numpy(), w.imag.numpy(), color, linewidth=2)
            plt.plot(w[0].real.numpy(), w[0].imag.numpy(), color, markersize=25)
            plt.plot(w[-1].real.numpy(), w[-1].imag.numpy(), color, markersize=25)
        else:
            plt.plot(w.real.numpy(), w.imag.numpy(), color=color, linestyle=style, linewidth=2)
            plt.plot(w[0].real.numpy(), w[0].imag.numpy(), color=color, marker=".", markersize=25)
            plt.plot(w[-1].real.numpy(), w[-1].imag.numpy(), color=color, marker=".", markersize=25)
        plt.show()

    return w, pole, eps, p