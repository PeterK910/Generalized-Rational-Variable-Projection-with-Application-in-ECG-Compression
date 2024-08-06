"""
Last Modified: August 05, 2024.
Version 1.0.

compress - Compressing the signal using the algorithm in [1].

References:
[1] M. Karczewicz, M. Gabbouj, ECG data compression by spline
    approximation, Signal Processing, vol. 59, pp. 43-59, 1997                
            
Usage: 
    [s,knots,coeff,prd]=compress(sig,order,knot_limit,init_knot,show);

Input parameters:
    sig        : The data points are interpreted as (x,y) where y=sig and x=1:length(sig). 
    order      : order of the B-splines (degree+1).
    knot_limit : number of knots at the final iteration.
    init_knot  : The algorithm removes one knot at each iteration from 
                this initial knot sequence. The knot having the least
                MSE is removed at each step.
    show       : display the corresponding B-spline approximation at each
                step (true/false).

Output parameters:
    s     : least square fitting of B-spline to 'sig' at the final iteration.
    knots : final knot sequence.
    coeff : coefficients of the least square spline at the last iteration.
    prd   : percent root mean square difference of the approximation at
            the final iteration.

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

from B_spline.predict_mse import predict_mse
from B_spline.bspline_coeffs import bspline_coeffs
from B_spline.preserve import preserve


def compress(sig, order, prd_limit, init_knot, show):

    x = np.arange(1, len(sig) + 1)
    psig = preserve(x, sig)
    knots = init_knot
    not_knots = []
    fig, ax=plt.subplots(2, height_ratios=[1,4])
    c, prd, s, Rl, zl = bspline_coeffs(psig, knots, order)
    step = 1

    while prd < prd_limit:

        wpp = np.zeros(len(knots) - 2)
        for i in range(2, len(knots)):
            wpp[i - 2] = predict_mse(psig, s, c, order, knots, i)

        indd = np.argmin(wpp)
        not_knot = knots[indd+1]
        tt = knots.copy()        
        knots = np.delete(knots, indd + 1)
        
        c, prd, s, Rl, zl = bspline_coeffs(psig, knots, order, indd + 1, tt, Rl, zl, show)        

        if show:
            not_knots.append(not_knot)
            width = np.max(psig) - np.min(psig)
            bl = np.min(s) - 0.1 * width
            ax[0].clear()
            ax[0].plot(
                not_knots,
                np.ones(len(not_knots)) * bl,
                "bx",
                markersize=12,
                linewidth=2,
            )
            ax[0].plot(knots, np.ones(len(knots)) * bl, "r.", markersize=5)
            ax[1].legend(
                [
                    "Original signal",
                    f"CR: {100 * (len(c) + len(knots)) / len(sig):.0f}, PRD: {prd:.02f}%",
                    
                ], loc='lower right'
            )
            plt.title(f"Iteration {step}")
            plt.draw()
            plt.pause(0.001)
            step += 1

    """
    if show:
        plt.figure()
        c, prd, s, _, _ = bspline_coeffs(psig, knots, order)
        x=np.arange(len(psig))
        plt.plot(x, psig, "b", linewidth=4)
        plt.plot(x, s, "r", linewidth=2)
        width = np.max(s) - np.min(s)
        bl = np.min(s) - 0.1 * width
        plt.plot(
            not_knots,
            np.ones(len(not_knots)) * bl,
            "bx",
            markersize=12,
            linewidth=2,
        )
        plt.plot(
            knots,
            np.ones(len(knots)) * bl,
            "r.",
            markersize=12
        )
        plt.stem(knots-1, psig[knots-1], "r.")
        plt.legend(["Original signal", f"PRD: {prd:.02f}%"])
        plt.axis([0, len(sig), np.min(psig) - 0.2 * width, np.max(psig) + 0.1 * width])
        plt.title("Approximation")

        plt.show()
    """
    return s, knots, c, prd
