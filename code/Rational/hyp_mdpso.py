"""
    Last Modified: July 01, 2024.
    Version 1.0.

    HYP_MDPSO  - Gives the pole configuration of the continuous MT system 
                that best fits the approximation of the function 'f'. This
                is a hyperbolic extension of the MD PSO algorithm [1].

    Usage: 
        [p,c,m,dbest,l,bl,prd]=hyp_mdpso_cr_vwin(f,ps_name,s,alpha,iterno,eps,show,insparts);

    Input parameters:
        f        : signal which is given as a ROW vector.  
        ps       : filename containing the hashtable which describes. 
                the spaces of the poles (see e.g., 'polespace1-3').
        s        : number of overall particles.
        alpha    : lies in [0,1] and controls the fitness function: alpha*PRD+(1-alpha)*CR.
        iterno   : number of iteration of the PSO algorithm.
        eps      : 2x2 matrix which contains the number of bits used for quantizing 
                the abs values and angles of the poles and coefficients:
                                | PolesAbsBit  PolesAngBit |
                                | CoeffAbsBit  CoeffAngBit |

        show     : optional logical value to display each step of the optimization process
        insparts : optional struct array to insert particles to the initial swarm:
                insparts(i).dim=inserted dimension number from [Dmin,Dmax] related to the 'ps' structure
                insparts(i).poles=inserted poles (length(poles)=ps{dim}) 

    Output parameters:
        p     : predicted poles of the continuous MT system with 'm' multiplicities
        c     : the Fourier coefficients of 'f' with respect to the continuous 
                MT system defined by the predicted poles 'p' 
        m     : multiplicites of the best pole configuration 'p'
        dbest : best dimension found by the optimization process
        l     : length of the approximated signal 
        bl:   : the signal 'f' is normalized by subtracting the base line, 
                so 'bl' contains the endpoints of the original signal which is
                neccessary for the reconstruction.
        prd   : approximation error in terms of PRD=(norm(sig-aprx)/norm(sig-mean(sig)))*100;


    The basic MDPSO algorithm was implemented by using the following article:
    [1] S. Kiranyaz,  J. Pulkkinen,  A. Yildirim,  and M. Gabbouj,  
        Multi-dimensional particle swarm optimization in dynamic environments, 
        Expert Systems with Applications, vol. 38, no. 3, pp. 2212-2223, 2011.

    [2] P. Kovács, S. Kiranyaz, M. Gabbouj, "Hyperbolic particle swarm optimization 
        with application in rational identification", Proceedings of the 21st European
        Signal Processing Conference (EUSIPCO), pp. 1-5, 2013. 

    Note: the same terminology and notations of paper [1] were used in this
        implementation. Additionally, the original MDPSO algorithm was 
        modified by using [2] to adapt the method to pole optimization.


    Copyright (c) 2017, Péter Kovács <kovika@inf.elte.hu>  
    Eötvös Loránd University, Budapest, Hungary, 2017.   

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
import random
from Rational.rait import addimag

def hyp_mdpso(f, ps_name, s, alpha = 0.5, iterno = 50, eps = np.array([8,8],[8,8]), show = False, insparts = {}):
    
    
    
    
    
    p=None
    c=None
    m=None
    dbest=None
    l=None
    bl=None
    prd=None    
    return p, c, m, dbest, l, bl, prd