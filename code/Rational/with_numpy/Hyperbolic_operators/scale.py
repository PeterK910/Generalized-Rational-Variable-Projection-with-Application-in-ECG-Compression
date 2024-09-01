from Rational.with_numpy.Hyperbolic_operators.section import section
from Rational.with_numpy.Hyperbolic_operators.eBlaschke import eBlaschke
import matplotlib.pyplot as plt
import numpy as np

def custom_atanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))

def scale(w1, w2, beta, draw, color=[0, 0, 0]):
    if w1 == w2:
        return 0, 0
    else:
        w, pole, eps, p = section(w1, w2, 100, draw, color)
        s05 = np.tanh(beta * custom_atanh(p))
        w05 = eBlaschke(pole, eps, s05)
        if draw:
            if isinstance(color, str):
                plt.plot(w05, color, markersize=10)
            else:
                plt.plot(w05, color, marker=".", markersize=10)

        return s05, w05
