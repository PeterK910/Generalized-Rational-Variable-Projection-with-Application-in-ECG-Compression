from Rational.Hyperbolic_operators.eBlaschke import eBlaschke
import matplotlib.pyplot as plt
import numpy as np

def section(w1, w2, resolution, draw, color=[0, 0, 0], style='-'):
    
    W = eBlaschke(w1, 1, w2)
    eps = W / np.abs(W)
    pole = -np.conj(eps) * w1
    p = np.abs(W)
    t = np.linspace(0, p, resolution)
    w = eBlaschke(pole, eps, t)

    if draw:
        
        if isinstance(color, str):
            plt.plot(w.real, w.imag, color, linewidth=2)
            plt.plot(w[0].real, w[0].imag, color, markersize=25)
            plt.plot(w[-1].real, w[-1].imag, color, markersize=25)
        else:
            plt.plot(w.real, w.imag, color=color, linestyle=style, linewidth=2)
            plt.plot(w[0].real, w[0].imag, color=color, marker='.', markersize=25)
            plt.plot(w[-1].real, w[-1].imag, color=color, marker='.', markersize=25)
        plt.show()

    return w, pole, eps.real, p