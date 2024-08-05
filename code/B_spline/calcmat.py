"""
    Last Modified: August 05, 2024.
    Version 1.0.

    calcmat - Computes the values of all the B-splines defined on the given knot sequence.
            The program constructs the matrix 'E' that contain the uniformly sampled ith 
            B-spline curve in the ith column. See Eq.(36) in [1].           

    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.    

    Usage: 
        E=calcmat(k,x,t);

    Input parameters:
        k    : order of the B-splines (degree+1). 
        x    : the B-splines are evaluated at points of 'x'.
        t    : vector of knots that defines the B-spline functions.

    Output parameters:
        E   : it is the matrix in the overdetermined systems of linear equations 
            of the corresponding least square problem. E(i,:) contains the values of the 
            ith B-spline curve for which the support is equal to the interval [t(i),t(i+k)].

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

def calcmat(k, x, t):
    # Discrete grid + boundary conditions
    tt = np.concatenate(([t[0]] * (k - 1), t, [t[-1]] * (k - 1)))
    n = len(x) - 2
    m = len(t) - 2
    E = np.zeros((n, m + k - 2))
    for i in range(m + k - 2):
        E[:, i] = Bspline(k - 1, i + 2, tt, x[1:-1]).T
        """
        Uncomment the following lines for plotting each Bspline
        
        plt.plot(np.linspace(x[0], x[-1], 100), Bspline(k - 1, i + 1, tt, np.linspace(x[0], x[-1], 100)))
        plt.draw()
        plt.pause(0.01)
        """
        
    return E
