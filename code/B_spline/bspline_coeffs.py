"""
    Last Modified: September 18, 2017.
    Version 1.0.

    bspline_coeffs - Computes the least sequare fitting of B-splines to discrete time signals.
                - This implementation is optimized for the compression algorithm in [1], i.e.
                    the matrices Rl, zl of the previous iteration of the compression can be passed 
                    to speed up the computation. See Eqs.(40)-(49) in [1].

    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.                
    
    Usage: 
        [c,mse,s,Rll,zll]=bspline_coeffs(sig,knots,order,p,tt,Rl,zl,show);

    Input parameters:
        sig   : data points that will be approximated by linear combination of B-splines. 
                The data points are interpreted as (x,y) where y=sig and x=1:length(sig). 
        knots : vector of knots that defines the B-splines.
        order : order of the B-spline functions (degree+1).
        p     : index of the knot that was removed in the previous iteration (optional).
        tt    : knot vector of the previous iteration.
        Rl,zl : auxiliary matrices for speeding up the computation. See Eqs.(40)-(49) in [1].
        show  : if it is true, the program displays the fitted curve (default=false).

    Output parameters:
        c     : coefficients of the least square fit.
        mse   : mean square error of the approximation.
        s     : values of the fitted curve at x=1:length(sig).
        Rl,zl : auxiliary matrices for the next iteration. See Eqs.(40)-(49) in [1].

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
from scipy.linalg import qr, solve_triangular
from B_spline.preserve import preserve
from B_spline.calcmat import calcmat
from B_spline.calcmat_opt import calcmat_opt
from B_spline.Bspline import Bspline


def bspline_coeffs(sig, knots, order, p=None, tt=None, Rl=None, zl=None, show=False):
    x = np.arange(1, len(sig) + 1)
    f = preserve(x, sig)
    y = preserve(x, f).reshape(-1, 1)

    if p is None:
        E = calcmat(order, x, knots)
        Qll, Rll = qr(E, mode='economic')
        zll = Qll.T @ y[1:-1]
        c = solve_triangular(Rll, zll)
        show = False
    else:
        B = calcmat_opt(order, x, tt, p)
        c, Rll, zll = solve_opt(Rl, zl, B, y[1:-1], order, p)

    s = np.zeros(len(sig))
    tt = np.concatenate([np.full(order - 1, knots[0]), knots, np.full(order - 1, knots[-1])])
    c = np.concatenate([[0], c, [0]])
    for i in range(len(c)):
        s += c[i] * Bspline(order - 1, i, tt, x)

    mse = np.sum((f - s) ** 2) / len(f)
    prd = np.sqrt(np.sum((f - s) ** 2) / np.sum((f - np.mean(f)) ** 2)) * 100
    mse = prd

    if show:
        plt.plot(f, 'b', linewidth=4)
        plt.plot(s, 'r', linewidth=2)
        plt.show()

    return c, mse, s, Rll, zll
