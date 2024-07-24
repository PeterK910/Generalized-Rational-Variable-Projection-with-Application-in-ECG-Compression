"""
    Last Modified: September 18, 2017.
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

def calcmat_opt(k, x, t, pp, El=None):
    m = len(t) - 2
    p = pp - 1
    if p < 0 or p > m:
        raise ValueError('The first and the last knot must be kept!')
    
    # Discrete grid + boundary conditions
    tt = np.concatenate(([t[0]] * (k - 1), t, [t[-1]] * (k - 1)))
    
    I1 = np.ones(p - 1)
    I2 = np.ones(m - p)
    lambda_ = (t[pp - 1] - tt[p-1:p + k-1]) / (tt[p-1 + k:p + 2 * k-1] - tt[p-1:p + k-1])
    mu = (tt[p + k :p + 2 * k ] - t[pp - 1]) / (tt[p + k :p + 2 * k ] - tt[p :p + k])
    
    i = np.zeros(m + 2 * k - 1, dtype=int)
    j = np.zeros(m + 2 * k - 1, dtype=int)
    s = np.zeros(m + 2 * k - 1)
    # Constructing the upper left corner (I1) of the 'B' matrix
    i[:p - 1] = np.arange(1, p)
    j[:p - 1] = np.arange(1, p)
    s[:p - 1] = I1
    # Constructing the middle part (B22) of the 'B' matrix
    i[p - 1:p + k - 1] = np.arange(p, p + k)
    j[p - 1:p + k - 1] = np.arange(p, p + k)
    s[p - 1:p + k - 1] = lambda_
    
    i[p + k - 1:p + 2 * k - 1] = np.arange(p + 1, p + k + 1)
    j[p + k - 1:p + 2 * k - 1] = np.arange(p, p + k)
    s[p + k - 1:p + 2 * k - 1] = mu
    
    # Constructing the bottom left corner (I2) of the 'B' matrix
    i[p + 2 * k - 1:] = np.arange(k + p + 1, m + k+1)
    j[p + 2 * k - 1:] = np.arange(k + p, m + k)
    s[p + 2 * k - 1:] = I2
    
    # Creating the sparse matrix
    B = csr_matrix((s, (i, j)), shape=(m + k + 1, m + k ))
    B = B[1:-2, 1:-2]
    
    if El is not None:
        E = El @ B
        return B, E
    return B

"""
# Example
k = 4
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475])
t = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255,257,259,261,263,265,267,269,271,273,275,277,279,281,283,285,287,289,291,293,295,297,299,301,303,305,307,309,311,313,315,317,319,321,323,325,327,329,331,333,335,337,339,341,343,345,347,349,351,353,355,357,359,361,363,365,367,369,371,373,375,377,379,381,383,385,387,389,391,393,395,397,399,401,403,405,407,409,411,413,415,417,419,421,423,425,427,429,431,433,435,437,439,441,443,445,447,449,451,453,455,457,459,461,463,465,467,469,471,473,475])
pp = 173

B = calcmat_opt(k, x, t, pp)
str=np.array2string(B, np.inf)
print(str)
#print("E:\n", E)

"""