import numpy as np

def genfilt6(theta1, theta2):
    # length 6 orthogonal scaling filter as a function of 2 angles.
    # Peter Kovacs, 2017
    # translated to python in 2024

    # One can get back db3 filter coefficients by using:
    # theta1=1.35980373244182; theta2=-0.78210638474440
    L = 6
    h = np.zeros(L)
    g = np.zeros(L)
    
    # Computing i=0,1th coefficients.
    for i in range(2):
        h[i] = (1 + (-1)**i * np.cos(theta1) + np.sin(theta1)) * \
               (1 - (-1)**i * np.cos(theta2) - np.sin(theta2)) + \
               ((-1)**i * 2 * np.sin(theta2) * np.cos(theta1))
        h[i] = h[i] / (4 * np.sqrt(2))
    
    # Computing i=2,3th coefficients.
    for i in range(2, 4):
        h[i] = (1 + np.cos(theta1 - theta2) + (-1)**i * np.sin(theta1 - theta2))
        h[i] = h[i] / (2 * np.sqrt(2))
    
    # Computing i=4,5th coefficients.
    for i in range(4, 6):
        h[i] = 1/np.sqrt(2) - h[i - 4] - h[i - 2]
    
    # Computing the coefficients of the wavelet filter 'g'.
    for n in range(L):
        g[n] = (-1)**n * h[L - n - 1]
    
    Lo_R = h
    Hi_R = g
    Lo_D = h[::-1]
    Hi_D = g[::-1]
    
    return Lo_D, Hi_D, Lo_R, Hi_R
