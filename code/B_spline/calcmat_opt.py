"""
    Last Modified: August 5, 2024.
    Version 1.0.
    
    calcmat_opt - Do the same as 'calcmat' but in a faster and more efficient way.
                It is an auxiliary function for speeding up the computation of the least 
                square spline in the next iteration. See Eqs.(41)-(45) in [1].
    
    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.                
                
    
    Usage: 
        [B,E]=calcmat_opt(k,x,t,pp,El); 
    
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
from scipy.sparse import csr_matrix


def calcmat_opt(k, t, pp, El=None):
    m = len(t) - 2
    p = pp - 1
    if p < 0 or p > m:
        raise ValueError("The first and the last knot must be kept!")

    # Discrete grid + boundary conditions
    tt = np.concatenate(([t[0]] * (k - 1), t, [t[-1]] * (k - 1)))

    if p == 0:
        I1 = np.array([])
    else:
        I1 = np.ones(p - 1)
    I2 = np.ones(m - p)

    lambda_ = (t[pp] - tt[p : p + k]) / (tt[p + k : p + 2 * k] - tt[p : p + k])
    mu = (tt[p + k + 1 : p + 2 * k + 1] - t[pp]) / (tt[p + k + 1 : p + 2 * k + 1] - tt[p + 1 : p + k + 1])

    i = np.zeros(m + 2 * k, dtype=int)
    j = np.zeros(m + 2 * k, dtype=int)
    s = np.zeros(m + 2 * k)
    # Constructing the upper left corner (I1) of the 'B' matrix
    if p > 0:
        i[: p - 1] = np.arange(1, p)
        j[: p - 1] = np.arange(1, p)
        s[: p - 1] = I1

    # Constructing the middle part (B22) of the 'B' matrix
    i[p : p + k] = np.arange(p, p + k)
    j[p : p + k] = np.arange(p, p + k)
    s[p : p + k] = lambda_

    i[p + k : p + 2 * k] = np.arange(p + 1, p + k + 1)
    j[p + k : p + 2 * k] = np.arange(p, p + k)
    s[p + k : p + 2 * k] = mu

    # Constructing the bottom left corner (I2) of the 'B' matrix
    i[p + 2 * k :] = np.arange(k + p + 1, m + k + 1)
    j[p + 2 * k :] = np.arange(k + p, m + k)
    s[p + 2 * k :] = I2

    # Creating the sparse matrix
    B = csr_matrix((s, (i, j)), shape=(m + k + 1, m + k))
    B = B[1:-2, 1:-2]

    if El is not None:
        E = El @ B
        return B, E
    return B
