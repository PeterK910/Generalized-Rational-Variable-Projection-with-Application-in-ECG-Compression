import numpy as np
import matplotlib.pyplot as plt
from Adaptive_Wavelets.genfilt6 import genfilt6
from Adaptive_Wavelets.matlab_wavelet_functions import wavedec
from Adaptive_Wavelets.matlab_wavelet_functions import waverec


def opt_params_wt(beat, PRD_limit, depth, acc, show, params):
    # Quantizing the parameters.
    tq = np.pi / (2**(acc[0]-1) - 1)
    theta1 = round(params[0] / tq)
    theta2 = round(params[1] / tq)
    theta1 = dequant(theta1, tq)
    theta2 = dequant(theta2, tq)
    # theta1=1.35980373244182
    # theta2=-0.78210638474440

    # Generate wavelet filter coefficients
    Lo_D, Hi_D, Lo_R, Hi_R = genfilt6(theta1, theta2)
    C, L = wavedec(beat, depth, Lo_D, Hi_D)
    sortedC = np.sort(np.abs(C))[::-1]
    indC = np.argsort(np.abs(C))[::-1]
    
    PRD = float('inf')
    CR = float('inf')
    Cths = np.copy(C)
    
    for K in range(0, len(C), 5):
        Cths = np.copy(C)
        Cths[indC[K:]] = np.zeros(len(Cths) - K)
        #Cths[indC[K:]] = np.zeros((len(Cths) - K, 1, 1))
        Cths, q = quant(Cths, acc[0])
        Cths = dequant(Cths, q)
        #print(Cths)
        #print(L)
        aprx = waverec(Cths, L, Lo_R, Hi_R)
        #print(aprx)
        #print(K, '---------------')
        #exit(0)
        PRD = np.linalg.norm(beat - aprx) / np.linalg.norm(beat - np.mean(beat)) * 100
        CR = K / len(Cths) * 100
        if PRD <= PRD_limit or K==50:
            break
    
    if show:
        x = np.arange(len(beat))
        plt.plot(x, beat, 'b', x, aprx, 'r', linewidth=2)
        h = plt.legend([f'PRD: {PRD:.2f} %', f'CR: {CR:.0f}%'])
        plt.setp(h, fontsize=15)
        plt.axis([0, len(beat), min(beat), max(beat)])
        plt.draw()
        plt.show()

    return CR, PRD, Cths, q, L


# AUXILIARY FUNCTIONS

def quant(data, eps):
    q = np.max(np.abs(data)) / (2**(eps-1) - 1)
    if q == 0:
        quantdata = np.ones_like(data)
    else:
        quantdata = np.round(data / q)
    return quantdata, q

def dequant(data, q):
    return data * q