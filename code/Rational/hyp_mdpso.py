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
from Rational.norm_sig import norm_sig
from Rational.rait.addimag import addimag
from Rational.rait.mt_coeffs import mt_coeffs
from Rational.rait.mt_system import mt_system
from Rational.rait.mt_generate import mt_generate
from Rational.rait.multiply_poles import multiply_poles
from Rational.rait.periodize_poles import periodize_poles
from Rational.Hyperbolic_operators.scale import scale
from Rational.Hyperbolic_operators.rho import rho
import numpy as np
import scipy.io as sio


def hyp_mdpso(f, ps_name, s, alpha = 0.5, iterno = 50, eps = None, show = False, insparts = []):
    
    if eps is None:
        eps = 8 * np.ones((2, 2))

    # Loading polespace
    if isinstance(ps_name, list):
        Dmin = 1
        Dmax = len(ps_name)
        ps = ps_name
    else:
        data = sio.loadmat(ps_name)
        ps = data['ps']
        ps=ps[0]
        Dmin = data['Dmin']
        Dmin = Dmin[0][0]
        Dmax = data['Dmax']
        Dmax = Dmax[0][0]

    # Initializing the swarm
    hf = addimag(f)
    xd = np.random.randint(1, Dmax+1, s)
    vd = np.random.randint(1, Dmax+1, s)
    xd_ = xd.copy()
    gbest = np.ones(Dmax, dtype=int)

    # Sorting user defined inserted particles
    if insparts:
        if len(insparts) > s:
            raise ValueError('Inserted particles must be smaller than swarm size.')
        else:
            insparts = convert2ps(insparts, Dmax)
    else:
        insparts = [None] * Dmax

    # Initializing the compression ratio for each dimension
    cr = np.ones(len(ps))
    for i in range(len(cr)):
        cn = np.sum(ps[i]) # number of complex coefficients
        pn = len(ps[i])
        cr[i] = (2 * (cn + pn) / len(f)) * 100

    # Step 1: Initializing positions and velocities
    xx = [None] * Dmax
    vx = [None] * Dmax
    xy = [None] * Dmax
    xy_g = [None] * Dmax
    pbesterr_a = [None] * Dmax
    gbesterr_d = [None] * Dmax
    print(gbesterr_d)

    for d in range(Dmin, Dmax+1):
        dim = 2 * len(ps[d-1])
        xx[d-1] = np.zeros((dim, s))
        vx[d-1] = np.zeros((dim, s))

        r = np.random.rand(dim//2, s)
        phi = 2 * np.pi * np.random.rand(dim//2, s)
        init_pole = r * np.exp(1j * phi)
        xx[d-1][0::2, :] = np.real(init_pole)
        xx[d-1][1::2, :] = np.imag(init_pole)

        r = np.random.rand(dim//2, s)
        phi = 2 * np.pi * np.random.rand(dim//2, s)
        init_vel = r * np.exp(1j * phi)
        vx[d-1][0::2, :] = np.real(init_vel)
        vx[d-1][1::2, :] = np.imag(init_vel)

        if insparts[d-1] is not None:
            insdim = insparts[d-1].shape[1]
            xx[d-1][:, :insdim] = insparts[d-1]

        err = errors_d(hf, xx[d-1], ps[d-1], alpha, eps)
        xy[d-1] = xx[d-1].copy()
        gbest[d-1] = np.argmin(err)
        xy_g[d-1] = xx[d-1][:, gbest[d-1]].copy()
        pbesterr_a[d-1] = err
        gbesterr_d[d-1] = err[gbest[d-1]]

    err = errors_gd(hf, xy_g, ps, s, alpha, eps)
    dbest = np.argmin(err) + 1

    # Initializing parameters for MDPSO
    c1 = 1.5
    c2 = 2
    w_high = 0.8
    w_low = 0.2
    w = w_high
    r = np.random.rand(2, 1)
    r1 = np.random.rand(Dmax, 1)
    r2 = np.random.rand(Dmax, 1)
    Vmax = 0.5
    VDmin = 0
    VDmax = 3
    maxPD = 8

    maxpn = len(ps[-1]) - 1
    for t in range(iterno):
        err = errors_xd(hf, xx, xd, ps, s, alpha, eps)
        for a in range(s):
            if err[a] < pbesterr_a[xd[a]-1][a]:
                xy[xd[a]-1][:, a] = xx[xd[a]-1][:, a].copy()
                pbesterr_a[xd[a]-1][a] = err[a]
                if err[a] < pbesterr_a[xd_[a]-1][a]:
                    xd_[a] = xd[a]

            ind = xd == xd[a]
            ind[a] = 0
            err_d = err[ind]

            if err[a] < min([gbesterr_d[xd[a]-1], min(err_d)]):
                gbest[xd[a]-1] = a
                xy_g[xd[a]-1] = xx[xd[a]-1][:, a].copy()
                gbesterr_d[xd[a]-1] = err[a]
                if err[a] < gbesterr_d[dbest-1]:
                    dbest = xd[a]

        for m in range(maxpn):
            for a in range(s):
                if m < len(ps[xd[a]-1]):
                    cy = array2complex(xy[xd[a]-1][2*m:2*m+2, a])
                    cx = array2complex(xx[xd[a]-1][2*m:2*m+2, a])
                    cy_ = array2complex(xy_g[xd[a]-1][2*m:2*m+2])
                    cv = array2complex(vx[xd[a]-1][2*m:2*m+2, a])

                    _, term1 = scale(0, cv, w, 0)
                    _, term2 = scale(0, add(-cx, cy), c1*r1[xd[a]-1], 0)
                    _, term3 = scale(0, add(-cx, cy_), c2*r2[xd[a]-1], 0)
                    term123 = add(add(term1, term2), term3)
                    if rho(0, term123) > Vmax:
                        _, term123 = scale(0, term123, 0.5/rho(0, term123), 0)
                    vx[xd[a]-1][2*m:2*m+2, a] = complex2array(term123)
                    xx[xd[a]-1][2*m:2*m+2, a] = complex2array(add(cx, term123))

                    vd[a] = np.floor(vd[a] + c1*r[0]*(xd_[a]-xd[a]) + c2*r[1]*(dbest-xd[a]))
                    xd_next = xd[a] + Cvd(vd[a], VDmin, VDmax)
                    PD = np.sum(xd == xd_next)
                    xd[a] = Cxd(xd[a], xd_next, Dmin, Dmax, PD, maxPD)

            r = np.random.rand(2, 1)
            r1 = np.random.rand(Dmax, 1)
            r2 = np.random.rand(Dmax, 1)

        w = w_high - t * (w_high - w_low) / iterno

        if show:
            db = dbest
            gbest_coords = xy_g[db-1]
            mult = ps[db-1]
            seg = f
            len_f = len(f)
            
            seg, base_line = norm_sig(seg)
            hseg = addimag(seg)
            period = 1
            tt = np.linspace(0, 2*np.pi, len(f) + 1)[:-1]
            
            print(f'Dbest dimension: {dbest}')
            print(f'Dbest multiplicities: {mult}')
            print(f'Dbest length: {len_f}')
            
            sz = np.zeros_like(xx[db-1])
            top_sz = 0
            for i in range(xx[db-1].shape[1]):
                if db == xd[i]:
                    sz[:, top_sz] = xx[db-1][:, i]
                    top_sz += 1
            sz = array2complex(sz[:, :top_sz]).T

            mpoles = periodize_poles(multiply_poles(array2complex(gbest_coords).T, mult), period)
            mpoles_r = quant(mpoles, 'pole', eps)

            c = mt_coeffs(hseg, mpoles)
            c_r = mt_coeffs(hseg, mpoles_r)
            c_r = quant(c_r, 'coeff', eps)

            fs_r = mt_generate(len_f, mpoles_r, c_r)
            prd_r = 100 * np.sqrt(np.sum((seg - np.real(fs_r))**2) / np.sum((seg - np.mean(seg))**2))
            fs = mt_generate(len_f, mpoles, c)
            prd = 100 * np.sqrt(np.sum((seg - np.real(fs))**2) / np.sum((seg - np.mean(seg))**2))
            cn = np.sum(mult)
            pn = len(mult)
            cr = 2 * (cn + pn) / len(f) * 100

            print(f'PRD: {prd:.2f}%')
            print(f'PRD_r: {prd_r:.2f}%')
            print(f'Compression ratio: {cr:.2f}%')
            print(f'Max pole distance: {np.max(np.abs(np.abs(mpoles) - 1))}')
            print(f'Number of poles: {len(mpoles)}')

    return xy_g, gbesterr_d, cr

# Computing the error function of a particle.
def error(hf, x, ps, alpha, eps):
    period = 1
    f = np.real(hf)
    err = np.zeros(x.shape[1])
    mult = ps[0]
    print("mult", mult)
    length = len(f)
    seg = f.copy() # Segmenting 'f' into smaller partitions.

    # Subtracting the baseline
    seg, base_line = norm_sig(seg)
    hseg = addimag(seg)
    print(x)
    for i in range(x.shape[1]):
        print("--",array2complex(x[:, i]).T)
        print("--",mult)
        poles = periodize_poles(multiply_poles(array2complex(x[:, i]).T, mult), period)
        
        # Quantizing the poles
        poles = quant(poles, 'pole', eps)
        
        # Computing the coefficients
        mts = mt_system(length, poles)
        co = (mts @ hseg.T / length).T
        
        # Quantizing the coefficients
        co = quant(co, 'coeff', eps)
        
        # Computing the percentage root mean square difference (PRD)
        aprx = np.real(co @ mts)
        prd = 100 * np.sqrt(np.sum((seg - aprx) ** 2) / np.sum((seg - np.mean(seg)) ** 2))
        
        # Computing the compression ratio
        cn = np.sum(mult)
        pn = len(mult)
        cr = (2 * (cn + pn) / length) * 100
        
        err[i] = alpha * prd + (1 - alpha) * cr
    
    return err
# Converting inserted particles from struct array to cell array that is compatible with the polespace
def convert2ps(insparts, Dmax):
    inspos = [None] * Dmax
    for i in range(1, Dmax + 1):
        currentdim = [part for part in insparts if part['dim'] == i]
        if currentdim:
            polenum = currentdim[0]['poles'].shape[1]
            inspoles = np.hstack([part['poles'] for part in currentdim])
            inspos[i - 1] = np.zeros((2 * inspoles.shape[0], inspoles.shape[1]))
            inspos[i - 1][0::2, :] = np.real(inspoles)
            inspos[i - 1][1::2, :] = np.imag(inspoles)
    return inspos

# Computing the error function for all particles in their related dimensions (xd(a))
# Note: the particles may lie in different dimensions.
def errors_xd(hf, xx, xd, ps, s, alpha, eps):
    err = np.zeros(s)
    for a in range(s):
        err[a] = error(hf, xx[xd[a] - 1][:, a], ps[xd[a] - 1], alpha, eps)
    return err

# Computing the error function for all particles for a certain dimension 'd' (xx{d}, ps{d})
# Note: all particles has the same dimension.
def errors_d(hf, xx, ps, alpha, eps):
    return error(hf, xx, ps, alpha, eps)

# Computing the error function for all global best particles in each dimension
def errors_gd(hf, xy_g, ps, s, alpha, eps):
    err = np.zeros(len(ps))
    for d in range(len(ps)):
        err[d] = error(hf, xy_g[d], ps[d], alpha, eps)
    return err

# Generating r1 and r2 vectors for all dimensions
def rand_r1r2(ps):
    Dmax = len(ps)
    r1 = [None] * Dmax
    r2 = [None] * Dmax
    for i in range(Dmax):
        d = len(ps[i]) * 2
        r1[i] = np.random.rand(d, 1)
        r2[i] = np.random.rand(d, 1)
    return r1, r2

# Clamping operator for dimension velocities
def Cvd(vd, VDmin, VDmax):
    if vd < VDmin:
        return VDmin
    elif vd > VDmax:
        return VDmax
    else:
        return vd

# Clamping operator for dimensions
def Cxd(xd_prev, xd_next, Dmin, Dmax, PD, maxPD):
    if PD >= maxPD or xd_next < Dmin or xd_next > Dmax:
        return xd_prev
    else:
        return xd_next

# Converting an array of real numbers to an array of complex numbers
def array2complex(ar):
    if len(ar.shape)==1:
        ar=np.reshape(ar, (ar.shape[0], 1))
    print("shape",(ar.shape))

    z = np.zeros((ar.shape[0] // 2, ar.shape[1]), dtype=complex)
    for i in range(ar.shape[1]):
        for j in range(ar.shape[0] // 2):
            z[j, i] = ar[2 * j, i] + 1j * ar[2 * j + 1, i]
    return z

# Converting an array of complex numbers to an array of real numbers
def complex2array(z):
    ar = np.zeros((2 * len(z), 1))
    ar[0::2, 0] = np.real(z).flatten()
    ar[1::2, 0] = np.imag(z).flatten()
    return ar

# Computing the hyperbolic vector addition
def add(z1, z2):
    return (z1 + z2) / (1 + np.conj(z1) * z2)

# Quantizing the poles and the coefficients
def quant(data, roundmode, eps):
    if roundmode == 'pole':
        qr = 1 / (2 ** eps[0, 0] - 1)
        qphi = np.pi / (2 ** (eps[0, 1] - 1) - 1)
    else:
        qr = 1 / (2 ** eps[1, 0] - 1)
        qphi = np.pi / (2 ** (eps[1, 1] - 1) - 1)

    # Quantizing the angles
    data_phi = np.round(np.angle(data) / qphi) * qphi

    # Quantizing the absolute values
    data_r = np.round(np.abs(data) / qr) * qr
    if roundmode == 'pole':
        data_r[data_r >= 1] = 1 - 1e-6  # Necessary to satisfy abs(poles) < 1

    return data_r * np.exp(1j * data_phi)
