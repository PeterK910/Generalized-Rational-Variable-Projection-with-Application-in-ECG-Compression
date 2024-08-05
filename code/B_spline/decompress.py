"""
Last Modified: August 05, 2024.
Version 1.0.

decompress - Decompressing the signal using the algorithm in [1].

References:
[1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
    approximation, Signal Processing, vol. 59, pp. 43-59, 1997                
            
Usage: 
    [s,prd]=decompress(sig,knots,coeff,order,bl,show);

Input parameters:
    sig        : The data points are interpreted as (x,y) where y=sig and x=1:length(sig). 
    knots      : knots of the B-spline functions.
    coeff      : coefficients of the least square spline approximation.
    order      : order of the B-splines (degree+1).
    bl         : According to Eqs. (31)-(32) in [1], the first and the
                last samples of the approximation are equal to zero, 
                i.e. s(1)=s(end)=0. In order to reconstruct the baseline 
                of the signal 'bl' contains the original values at the endpoints,
                i.e. bl=[sig(1),sig(end)].
    show       : display the corresponding B-spline approximation (true/false).

Output parameters:
    s     : least square fitting of B-spline to 'sig' at the final iteration.
    prd   : percent root mean square difference of the approximation to
            the original signal.

Copyright (c) 2017, P�ter Kov�cs <kovika@inf.elte.hu>  
E�tv�s Lor�nd University, Budapest, Hungary, 2017.   

Permission to use, copy, modify, and/or distribute this software for  
any purpose with or without fee is hereby granted, provided that the  
above copyright notice and this permission notice appear in all copies.  

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL  
WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED  
WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR  
BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES  
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS  
SOFTWARE. 
"""

import numpy as np
import matplotlib.pyplot as plt

from B_spline.Bspline import Bspline

def decompress(signal, knots, coeff, order, bl, show):
    x = np.arange(0, len(signal))
    s = np.zeros(len(signal))
    tt = np.concatenate((np.ones(order - 1) * knots[0], knots, np.ones(order - 1) * knots[-1]))

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


