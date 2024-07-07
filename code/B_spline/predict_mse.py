"""
    Last Modified: September 18, 2017.
    Version 1.0.

    predict_mse - Predicts the mean square error (MSE) of a removed knot.
                See Eqs.(63)-(65) in [1].

    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997                

    Usage: 
        wp=predict_mse(f,g,c,k,t,pp);

    Input parameters:
        f  : The data points are interpreted as (x,y) where y=f and x=1:length(f). 
        g  : Least square fitted spline curve to f. 
            It is a linear combination of B-splines defined on the knot vector t.
        c  : coefficients of the spline curve g.
        k  : degree of the B-splines (order-1).
        t  : vector of knots.
        pp : index of the removing knot for which the MSE is predicted.

    Output parameters:
        wp    : predicted MSE of the ppth knot.

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

from B_spline.Bspline import Bspline
import numpy as np

def predict_mse(f, g, c, k, t, pp):
    x = np.arange(1, len(f) + 1)
    m = len(t) - 2
    p = pp - 1
    if p < 1 or p > m:
        raise ValueError('The first and the last knot must be kept!')
    N = len(f)
    tt = np.concatenate([np.full(k - 1, t[0]), t, np.full(k - 1, t[-1])])
    rrho = np.concatenate([tt[:k - 1 + p], tt[k + 1 + p:]])
    alpha = (t[pp] - rrho[p:p + k - 1]) / (rrho[p + k:p + 2 * (k - 1)] - rrho[p:p + k - 1])

    d = np.zeros(m + k - 1)
    d[:p + 1] = c[:p + 1]
    d[p + k - 1:] = c[p + k:]

    for i in range(p + 1, p + k - 1):
        d[i] = (c[i] - (1 - alpha[i - p]) * d[i - 1]) / alpha[i - p]

    dd = alpha[-1] * d[p - 1 + k] + (1 - alpha[-1]) * d[p - 1 + k - 1]

    dzeta1 = (f - g) + (c[p - 1 + k] - dd) * Bspline(k - 1, p - 1 + k, tt, x)
    dzeta1 = np.sum(dzeta1 ** 2) / N

    d = np.zeros(m + k - 1)
    d[:p + 1] = c[:p + 1]
    d[p + k - 1:] = c[p + k:]

    for i in range(p + k - 1, p + 1, -1):
        d[i - 1] = (c[i] - alpha[i - p] * d[i]) / (1 - alpha[i - p])

    dd = alpha[0] * d[p + 1] + (1 - alpha[0]) * d[p]

    dzeta2 = (f - g) + (c[p + 1] - dd) * Bspline(k - 1, p + 1, tt, x)
    dzeta2 = np.sum(dzeta2 ** 2) / N

    wp = min(dzeta1, dzeta2)
    return wp

