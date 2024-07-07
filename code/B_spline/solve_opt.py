"""
    Last Modified: September 18, 2017.
    Version 1.0.
    
    solve_opt - Auxiliary function for speeding up the computation of the least 
                square spline fitting in the next iteration. See Eq.(47) in [1].
    
    References:
    [1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
        approximation, Signal Processing, vol. 59, pp. 43-59, 1997.                
                
    
    Usage: 
        [c,Rll,zll]=solve_opt(Rl,zl,B,b,k,p); 
    
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
from scipy.linalg import solve_triangular

def solve_opt(Rl, zl, B, b, k, p):
    Rll = np.dot(Rl, B)
    zll = np.copy(zl)
    m, n = Rll.shape

    if p - 2 <= 0:
        ind = 0
    else:
        ind = p - 2

    def planerot(x):
        if x[1] == 0:
            G = np.eye(2)
            y = x
        else:
            r = np.hypot(x[0], x[1])
            c = x[0] / r
            s = -x[1] / r
            G = np.array([[c, s], [-s, c]])
            y = np.array([r, 0])
        return G, y

    for i in range(ind, n - k):
        G, y = planerot(Rll[i:i+2, i])
        zll[i:i+2] = np.dot(G, zll[i:i+2])
        Rll[i:i+2, i] = y
        for j in range(k - 1):
            r = np.dot(G, Rll[i:i+2, i + j + 1])
            Rll[i:i+2, i + j + 1] = r

    if n - k <= 0:
        ind = 0
    else:
        ind = n - k

    for i in range(ind, n):
        G, y = planerot(Rll[i:i+2, i])
        zll[i:i+2] = np.dot(G, zll[i:i+2])
        Rll[i:i+2, i] = y
        for j in range(n - i - 1):
            r = np.dot(G, Rll[i:i+2, i + j + 1])
            Rll[i:i+2, i + j + 1] = r

    c = solve_triangular(Rll[:n, :n], zll[:n], lower=False, unit_diagonal=False, overwrite_b=False, check_finite=True)
    return c, Rll, zll

# Example usage:
Rl = np.array([[1, 2], [3, 4]])
zl = np.array([5, 6])
B = np.array([[7, 8], [9, 10]])
b = 0
k = 1
p = 2

c, Rll, zll = solve_opt(Rl, zl, B, b, k, p)
print("c:", c)
print("Rll:", Rll)
print("zll:", zll)
