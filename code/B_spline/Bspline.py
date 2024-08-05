"""
    Last Modified: August 18, 2024.
    Version 1.0.

    Bspline - Computes the values of B-spline curves by 
            using the Cox-de Boor recursion formula. 

    Usage: 
        y=Bspline(l,k,xk,x);

    Input parameters:
        l : degree of the B-splines. 
        k : the program computes the values of the kth B-spline, 
            the support of this B-spline is the interval [xk(k),xk(k+l+1)]).
        xk: vector of knots that defines the B-spline functions.
        x : the B-splines are evaluated at 'x'.

    Output parameters:
        y : values of the B-spline at 'x'.

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


def Bspline(l, k, xk, x):
    if l == 0:
        y = np.ones_like(x) * (x >= xk[k-1]) * (x < xk[k])
    else:
        
        if (xk[k-1 + l] - xk[k-1]) != 0:
            y1 = Bspline(l - 1, k, xk, x) * (x - xk[k-1]) / (xk[k-1 + l] - xk[k-1])
        else:
            y1 = np.zeros_like(x)
        
        if (xk[k + l] - xk[k]) != 0:
            y2 = Bspline(l - 1, k+1, xk, x) * (xk[k + l] - x) / (xk[k + l ] - xk[k ])
        else:
            y2 = np.zeros_like(x)
        y = y1 + y2
    return y

