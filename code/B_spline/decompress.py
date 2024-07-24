import numpy as np
import matplotlib.pyplot as plt

from B_spline.Bspline import Bspline

def decompress(signal, knots, coeff, order, bl, show):
    #x = np.arange(1, len(signal) + 1)
    x = np.arange(0, len(signal) )
    s = np.zeros(len(signal))
    #print(order)
    #print(knots)
    tt = np.concatenate((np.ones(order - 1) * knots[0], knots, np.ones(order - 1) * knots[-1]))
    #print(tt)
    #print(tt.shape)
    # Bspline reconstruction
    for i in range(len(coeff)):
        p = coeff[i] * Bspline(order - 1, i + 1, tt, x)
        s += p
    # Baseline reconstruction
    slope = (bl[-1] - bl[0]) / (len(signal)-1)
    base_line = bl[0] + slope * (x - 1)
    s += base_line

    prd = np.sqrt(np.sum((signal - s) ** 2) / np.sum((signal - np.mean(signal)) ** 2)) * 100
    cr = (len(knots) + len(coeff) - 1) / len(signal) * 100
    #print(prd)
    #print(cr)
    if show:
        """
        plt.figure()
        plt.gca().set_aspect('auto', adjustable='box')
        plt.plot(x, signal, 'b', label='Original Signal')
        plt.plot(x, s, 'r', linewidth=2, label='Approximation')
        plt.stem(knots, signal[knots-1], 'r.')
        plt.grid(True)
        plt.legend()
        plt.axis('square')
        plt.ylim(-6, 6)
        plt.tight_layout()
        plt.show()
        """

    return s, prd


