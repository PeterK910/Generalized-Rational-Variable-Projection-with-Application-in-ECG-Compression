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
from B_spline.solve_opt import solve_opt
from B_spline.Bspline import Bspline


def bspline_coeffs(sig, knots, order, p=None, tt=None, Rl=None, zl=None, show=False):
    x = np.arange(1, len(sig) + 1)
    f = preserve(x, sig)
    y = preserve(x, f)

    if p is None:
        E = calcmat(order, x, knots)
        Qll, Rll = qr(E, mode='economic')
        zll = Qll.T @ y[1:-1]
        c = solve_triangular(Rll, zll)
        show = False
    else:
        B = calcmat_opt(order, x, tt, p).toarray()
        c, Rll, zll = solve_opt(Rl, zl, B, y[1:-1], order, p)

    s = np.zeros(len(sig))
    tt = np.concatenate([np.full(order - 1, knots[0]), knots, np.full(order - 1, knots[-1])])
    c = np.concatenate([[0], c, [0]])
    for i in range(len(c)):
        s += c[i-1] * Bspline(order - 1, i, tt, x)
    mse = np.sum((f - s) ** 2) / len(f)
    prd = np.sqrt(np.sum((f - s) ** 2) / np.sum((f - np.mean(f)) ** 2)) * 100
    mse = prd

    if show:
        # ax=plt.gca()
        plt.cla()
        plt.plot(f, 'b', linewidth=4)
        plt.plot(s, 'r', linewidth=2)
        plt.pause(0.01)
        """
        plt.show()
        """

    return c, mse, s, Rll, zll


"""
psig=np.array([0,-0.03049,-0.030165,-0.076061,-0.029514,-0.060004,0.032765,0.079312,0.079637,0.157,0.15732,0.12683,0.080937,0.096669,0.096994,0.081912,0.11305,0.12878,0.20615,0.12943,0.19139,0.20712,0.22285,0.19236,0.17728,0.1622,0.19334,0.16285,0.10155,0.14809,0.21005,0.19496,0.2261,0.24184,0.21135,0.25789,0.15037,0.13529,0.1048,0.074306,0.074631,0.028734,0.059874,0.091014,0.091339,0.13789,0.076581,0.061499,0.077232,0.062149,0.047067,0.031985,0.047717,0.094264,0.079182,0.12573,0.18768,0.15719,0.20374,0.20407,0.12735,0.11227,0.050968,0.035885,0.020803,0.051943,0.036861,0.068,0.068325,0.053243,0.038161,0.053893,0.023403,-0.022493,-0.006761,0.0089713,-0.06774,-0.0057859,-0.0054608,0.010272,-0.0048107,0.026329,0.057469,0.026979,0.027304,-0.034,-0.06449,-0.03335,-0.06384,-0.10974,-0.032375,-0.016642,-0.00091014,0.061044,0.045962,0.015472,-0.015017,0.00071511,-0.045182,-0.029449,-0.059939,0.0020153,0.048562,0.079702,0.14166,0.14198,0.15771,0.18885,0.21999,0.25113,0.31309,0.35963,0.51403,0.62221,0.68416,0.60745,0.37667,-0.039006,-0.5009,-0.96279,-1.4863,-1.9944,-2.5334,-3.2264,-3.9194,-4.5353,-5.1359,-5.5208,-5.6899,-5.8128,-5.8433,-5.8892,-5.8735,-5.8732,-5.8728,-5.8109,-5.6257,-5.3942,-5.0396,-4.7465,-4.4534,-4.145,-3.8981,-3.6205,-3.312,-3.1114,-2.9107,-2.7255,-2.4633,-2.2318,-1.9388,-1.6457,-1.4143,-1.0904,-0.7049,-0.39643,-0.11877,0.081847,0.15921,0.12872,0.15986,0.17559,0.28377,0.34572,0.48471,0.54667,0.67025,0.70139,0.65549,0.71745,0.65614,0.67188,0.70302,0.70334,0.76529,0.79643,0.90461,0.95116,0.93607,0.9364,0.98295,0.93705,0.95278,0.98392,0.99965,1.0462,1.0773,1.1855,1.2167,1.3402,1.3714,1.4487,1.4645,1.511,1.5422,1.5425,1.5736,1.7126,1.79,1.8827,1.9293,2.0375,2.1456,2.1614,2.2079,2.162,2.1778,2.1627,2.1322,2.1787,2.1945,2.2102,2.2722,2.3495,2.4577,2.5659,2.6278,2.5665,2.6439,2.6288,2.6908,2.7989,2.8455,2.892,3.031,3.0005,3.0471,2.9704,2.8936,2.8478,2.8019,2.756,2.7255,2.6642,2.6337,2.6032,2.5111,2.496,2.3422,2.2655,2.1426,2.0042,1.8967,1.743,1.62,1.4817,1.4204,1.3591,1.2362,1.1749,1.0365,0.9444,0.83687,0.74475,0.591,0.56051,0.46839,0.4225,0.36119,0.40774,0.36184,0.37758,0.34709,0.332,0.30152,0.28643,0.19431,0.17923,0.13333,0.16447,0.18021,0.21135,0.21167,0.28903,0.28936,0.25887,0.30542,0.24411,0.22903,0.21395,0.21427,0.18378,0.23033,0.23065,0.2772,0.30834,0.32407,0.30899,0.29391,0.27883,0.26374,0.18703,0.17195,0.18768,0.21882,0.24996,0.25029,0.29683,0.34338,0.31289,0.26699,0.25191,0.28305,0.19093,0.25289,0.19158,0.23813,0.28468,0.285,0.31614,0.30106,0.27057,0.2863,0.17878,0.16369,0.16402,0.17975,0.13385,0.14959,0.21154,0.22727,0.27382,0.25874,0.24366,0.21317,0.18268,0.19841,0.10629,0.10662,0.076126,0.10727,0.13841,0.13873,0.20068,0.1702,0.1243,0.10922,0.094134,0.094459,0.06397,0.048887,0.049212,0.095759,0.11149,0.12722,0.14296,0.12787,0.15901,0.12852,0.082627,0.082952,0.0062409,0.052788,0.022298,0.038031,0.06917,0.069495,0.10064,0.10096,0.0088413,0.070796,0.024899,-0.036405,-0.0052658,-0.0049407,-0.0046157,0.072746,0.057664,0.073396,0.073721,0.10486,0.10519,0.028474,0.013392,0.059939,-0.0013652,-0.016447,-0.03153,-0.00039006,-6.501e-05,0.046482,0.062214,0.031725,0.062864,0.032375,0.017293,0.017618,-0.012872,-0.012547,-0.012222,-0.011897,0.050057,0.081197,0.11234,0.11266,0.082172,0.082497,0.0366,0.021518,-0.024379,-0.039461,-0.06995,-0.023403,-0.038486,0.023469,0.054608,-0.006696,0.055258,0.055583,0.0096864,-0.0053958,0.010337,-0.066375,-0.0044207,-0.050318,0.027044,0.027369,0.058509,0.028019,0.028344,-0.0021453,-0.048042,-0.063124,-0.13984,-0.093289,-0.10837,-0.12345,-0.10772,0.00045507,0.016187,0.016512,0.032245,0.0017553,0.017488,-0.074631,-0.12053,-0.13561,-0.11988,-0.10415,-0.073006,-0.057274,-0.087763,-0.010402,-0.025484,-0.025159,-0.086463,-0.086138,-0.16285,-0.19334,-0.17761,-0.10024,-0.053698,0.0082562,0.023989,0.055128,0.0092314,0.071186,-0.020933,-0.06683,-0.11273,-0.096994,-0.065855,-0.034715,-0.049797,-0.0032505,0.043296,0.012807,-0.048497,0.013457,-0.047847,-0.047522,-0.12423,-0.1085,-0.092769,0])
knots=np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255,257,259,261,263,265,267,269,271,273,275,277,279,281,283,285,287,289,291,293,295,297,299,301,303,305,307,309,311,313,315,317,319,321,323,325,327,329,331,333,335,337,339,341,343,345,347,349,351,353,355,357,359,361,363,365,367,369,371,373,375,377,379,381,383,385,387,389,391,393,395,397,399,401,403,405,407,409,411,413,415,417,419,421,423,425,427,429,431,433,435,437,439,441,443,445,447,449,451,453,455,457,459,461,463,465,467,469,471,473,475])
order=4
c, prd, s, rl, zl=bspline_coeffs(psig,knots,order)
print(c)
print(prd)
print(s)
print(rl[1])
print(zl)
"""