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
import matplotlib.pyplot as plt


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
    #print(f )
    hf = addimag(f) # Computing the Hilbert-transform of 'f'.
    #print(hf )
    #exit(0)
    xd = np.random.randint(1, Dmax+1, s) # Initializing the particle's dimesions.
    vd = np.random.randint(1, Dmax+1, s) # Initializing the particles's dimension velocities.
    xd_ = xd.copy() # Initializing the personal best dimensions of the particles.
    gbest = np.ones(Dmax, dtype=int) # Initializing the gbest particle at each dimenions.
    #print(gbest)
    # Sorting user defined inserted particles
    if insparts:
        if len(insparts) > s:
            raise ValueError('Inserted particles must be smaller than swarm size.')
        else:
            insparts = convert2ps(insparts, Dmax)
    else:
        insparts = [None] * Dmax
    #print(insparts)
    # Initializing the compression ratio for each dimension
    cr = np.ones(len(ps))
    for i in range(len(cr)):
        cn = np.sum(ps[i]) # number of complex coefficients
        pn = len(ps[i])
        cr[i] = (2 * (cn + pn) / len(f)) * 100
    #print(ps)
    # Step 1: Initializing positions and velocities
    xx = [None] * Dmax # Positions of the particles.
    vx = [None] * Dmax # Velocities of the particles.
    xy = [None] * Dmax # Personal best positions of the particles.
    xy_g = [None] * Dmax # Global best position in each dimension.
    pbesterr_a = [None] * Dmax # Personal best errors of the particles.
    gbesterr_d = [None] * Dmax # Global best errors in each dimension.

    #print(Dmin, Dmax)
    for d in range(Dmin, Dmax+1):
        ##print("ps:",ps)
        dim = 2 * len(ps[d-1][0]) # The poles are complex numbers, so the total dimension of the problem is 2*num_poles.
        xx[d-1] = np.zeros((dim, s))
        vx[d-1] = np.zeros((dim, s))

        # Initilaizing positions
        r = np.random.rand(dim//2, s)
        phi = 2 * np.pi * np.random.rand(dim//2, s)
        init_pole = r * np.exp(1j * phi)
        xx[d-1][0::2, :] = np.real(init_pole)
        xx[d-1][1::2, :] = np.imag(init_pole)

        # Initilaizing velocities  
        r = np.random.rand(dim//2, s)
        phi = 2 * np.pi * np.random.rand(dim//2, s)
        init_vel = r * np.exp(1j * phi)
        vx[d-1][0::2, :] = np.real(init_vel)
        vx[d-1][1::2, :] = np.imag(init_vel)

        # Inserting user defined particles to the initial swarm.
        if insparts[d-1] is not None:
            insdim = insparts[d-1].shape[1] # Number of inserted particle at dimension 'd'.
            xx[d-1][:, :insdim] = insparts[d-1] # Overwrite the first insdim number of particles at dimension 'd'.

        ###print("ps",ps[d-1])
        #print(hf)
        err = errors_d(hf, xx[d-1], ps[d-1][0], alpha, eps)
        #print(eps)
        xy[d-1] = xx[d-1].copy()
        gbest[d-1] = np.argmin(err)
        xy_g[d-1] = xx[d-1][:, gbest[d-1]].copy()
        pbesterr_a[d-1] = err
        gbesterr_d[d-1] = err[gbest[d-1]]
    #print(gbesterr_d)
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
    fig, ax=plt.subplots(1, 2)

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
            #print(ind)
            #print(err)
            """
            if err[a] < min_2arg([gbesterr_d[xd[a]-1], min_1arg(err_d)]):
                gbest[xd[a]-1] = a
                xy_g[xd[a]-1] = xx[xd[a]-1][:, a].copy()
                gbesterr_d[xd[a]-1] = err[a]
                if err[a] < gbesterr_d[dbest-1]:
                    dbest = xd[a]
            """
            min_err_d = min(err_d) if len(err_d)>0 else float('inf')

            if err[a] < min(gbesterr_d[xd[a]-1], min_err_d):  # (Step 3.1.4.)
                gbest[xd[a]-1] = a
                xy_g[xd[a]-1] = xx[xd[a]-1][:, a].copy()
                gbesterr_d[xd[a]-1] = err[a]
                if err[a] < gbesterr_d[dbest-1]:  # (Step 3.1.4.2.)
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
            gbest_coords = xy_g[db]
            mult = ps[db][0]
            seg = f  # Segmenting 'f' into smaller partitions.
            len_f = len(f)
            
            # Subtracting the baseline.
            seg, base_line = norm_sig(seg)
            hseg = addimag(seg)
            hseg=np.reshape(hseg, (1, hseg.shape[0]))
            period = 1
            tt = np.linspace(0, 2 * np.pi, len(f) + 1)
            tt = tt[:len(f)]
            
            # Displaying the actual dbest dimension and multiplicities and lengths.
            print(f'Dbest dimension: {dbest}')
            print(f'Dbest multiplicities: {mult}')
            print(f'Dbest length: {len_f}')
            
            # Displaying the pole configuration of each particle in the dbest dimension.
            sz = np.zeros_like(xx[db])
            top_sz = 1
            for i in range(xx[db].shape[1]):
                if db == xd[i]:
                    sz[:, top_sz - 1] = xx[db][:, i]
                    top_sz += 1
            sz2=sz[:, :top_sz ]
            sz2=np.reshape(sz2, (sz2.size, 1))
            sz = array2complex(sz2).T

            # Calculating and quantizing the poles.
            mpoles = periodize_poles(multiply_poles(array2complex(gbest_coords).T, mult), period)
            mpoles_r = quant(mpoles, 'pole', eps)
            # Calculating the coefficients WITHOUT quantized poles.
            mpoles=np.reshape(mpoles, (1, mpoles.shape[0]))
            c = mt_coeffs(hseg, mpoles)
            #print(c)
            # Calculating the coefficients WITH quantized poles.
            mpoles_r=np.reshape(mpoles_r, (1, mpoles_r.shape[0]))
            c_r = mt_coeffs(hseg, mpoles_r)
            # Quantizing the coefficients.
            c_r = quant(c_r[0], 'coeff', eps)

            # Computing the error in terms of PRD.
            fs_r = mt_generate(len_f, mpoles_r, c_r)
            prd_r = 100 * np.sqrt(np.sum((seg - np.real(fs_r))**2) / np.sum((seg - np.mean(seg))**2))
            fs = mt_generate(len_f, mpoles, c[0])
            prd = 100 * np.sqrt(np.sum((seg - np.real(fs))**2) / np.sum((seg - np.mean(seg))**2))
            
            # Computing the compression ratio (CR).
            cn = np.sum(mult) * period  # number of complex coefficients
            pn = len(mult)
            cr = (2 * (cn + pn) / len_f) * 100

            ax[0].clear()
            
            #plt.subplots(1, 2)
            unit_disc = np.exp(1j * tt)
            #plt.plot(unit_disc, 'k')
            ax[0].plot(unit_disc, 'k')
            plt.title(f'step: {t}')
            
            styles = ['bo', 'bx', 'b.', 'b+', 'bs', 'bv', 'bp', 'bh']
            styles_best = ['ro', 'rx', 'r.', 'r+', 'rs', 'rv', 'rp', 'rh']
            
            for j in range(len(mult)):
                #plt.plot(sz[j], styles[j])
                ax[0].plot(sz[j], styles[j])
            
            # Plotting the global best pole configuration in the dbest dimension.
            # Note: sudden changes on this figure indicate changes in the dbest dimension.
            for i in range(len(mult)):
                #plt.plot(array2complex(gbest_coords[i*2-1:i*2]), styles_best[i], markersize=15, linewidth=4)
                ax[0].plot(array2complex(gbest_coords[i*2-1:i*2]), styles_best[i], markersize=15, linewidth=4)
            
            
            # Displaying the rational approximation of the segment.
            #plt.subplot(1, 2, 1)
            #plt.plot(tt, np.real(hf), 'b', linewidth=4)
            ax[1].plot(tt, np.real(hf), 'b', linewidth=4)

            # plt.plot(tt[:len_f], np.real(seg) + base_line, 'g', linewidth=3)
            # plt.plot(tt[:len_f], np.real(fs) + base_line, 'r', linewidth=3)
            ax[1].plot(tt[:len_f], np.real(fs_r)[0] + base_line, 'r', linewidth=1)
            #plt.plot(tt[:len_f], np.real(fs_r)[0] + base_line, 'r', linewidth=1)
            
            #plt.legend(['Original signal', f'CR: {len_f / (2 * (cn + pn))}:1, PRD: {prd_r}'])
            ax[1].legend(['Original signal', f'CR: {len_f / (2 * (cn + pn))}:1, PRD: {prd_r}'])
            #plt.axis('tight')
            ax[1].axis('tight')
            plt.show()
    
    # Return the gbest poles and the quantized coefficients of the dbest dimension.
    m = ps[dbest][0]
    poles = array2complex(xy_g[dbest])

    # Return dbest length.
    l = len(f)

    # Return the base_line.
    seg, bl = norm_sig(f[:l])
    hseg = addimag(seg)
    hseg = np.reshape(hseg, (1, len(hseg)))
    # Quantizing the poles.
    poles = quant(poles, 'pole', eps).T   
    p = poles
    mpoles = periodize_poles(multiply_poles(poles, m), 1)
    mpoles=np.reshape(mpoles, (1, len(mpoles)))

    # Calculating the coefficients WITH quantized poles.
    c = mt_coeffs(hseg, mpoles)

    # Quantizing the coefficients.
    c = quant(c[0], 'coeff', eps)
    #print(c)

    # Calculating the PRD for 'f'.
    aprx = np.real(mt_generate(l, mpoles, c))
    prd = 100 * np.sqrt(np.sum((seg - aprx) ** 2) / np.sum((seg - np.mean(seg)) ** 2))

    return p, c, m, dbest, l, bl, prd

# Computing the error function of a particle.
def error(hf, x, ps, alpha, eps):
    #print(hf)
    #print(x)
    #print(ps)
    #exit(0)
    #print(np.mean(hf))
    #print(np.mean(x))
    #print(np.mean(ps))
    
    period = 1
    f = np.real(hf)
    if len(x.shape)<2:
        x=np.reshape(x, (x.shape[0], 1))
    #print("->",x)
    
    err = np.zeros(x.shape[1])
    mult = ps.copy()
    ###print("mult", mult)
    length = len(f)
    seg = f.copy() # Segmenting 'f' into smaller partitions.

    # Subtracting the baseline
    seg, base_line = norm_sig(seg)
    #print(seg)
    hseg = addimag(seg)
    #print(hseg)
    #print("hseg:",np.max(hseg))
    #print(x.shape)
    #print(x)
    for i in range(x.shape[1]):
        #print(i,(x[:, i]))
        #print(i,array2complex(x[:, i]).T)
        if isinstance(mult[0], np.ndarray):
            mult=mult[0]
        poles = periodize_poles(multiply_poles(array2complex(x[:, i]).T, mult), period)
        #print("pole",poles)
        
        # Quantizing the poles
        poles = quant(poles, 'pole', eps)
        poles=np.reshape(poles, (1, poles.shape[0]))
        # Computing the coefficients
        mts = mt_system(length, poles)
        #print(hseg.T)
        co = (np.matmul(mts, hseg.T) / length).T
        #print(co)

        # Quantizing the coefficients
        co = quant(co, 'coeff', eps)
        
        # Computing the percentage root mean square difference (PRD)
        aprx = np.real(np.matmul(co, mts))
        #print(np.sqrt(np.sum((seg-aprx))**2))
        prd = 100 * np.sqrt(np.sum(np.power((seg - aprx), 2)) / np.sum(np.power((seg - np.mean(seg)), 2)))
        #print(seg - aprx)
        
        # Computing the compression ratio
        cn = np.sum(mult)
        pn = len(mult)
        cr = (2 * (cn + pn) / length) * 100
        
        err[i] = alpha * prd + (1 - alpha) * cr
        #print("co", err[i])
    #print(err)
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
    ###print(ar)
    ###print("shape",(ar.shape))

    z = np.zeros((ar.shape[0] // 2, ar.shape[1]), dtype=complex)
    ###print(z)
    for i in range(ar.shape[1]):
        for j in range(ar.shape[0] // 2):
            z[j, i] = ar[2 * j, i] + 1j * ar[2 * j + 1, i]
    #print(z)
    z=np.reshape(z, (1, z.shape[0]))
    ###print("ujforma:",z)
    return z[0]

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

hf=np.array([-2.9918e-16+0.090638j,-0.03049+0.04861j,-0.030165+0.087977j,-0.076061+0.022376j,-0.029514+0.013453j,-0.060004-0.029843j,0.032765-0.085191j,0.079312-0.028085j,0.079637-0.039103j,0.157-0.021716j,0.15732+0.061776j,0.12683+0.087167j,0.080937+0.091676j,0.096669+0.048425j,0.096994+0.085955j,0.081912+0.03898j,0.11305+0.049523j,0.12878+0.0069098j,0.20615+0.07401j,0.12943+0.090913j,0.19139+0.049894j,0.20712+0.10544j,0.22285+0.1295j,0.19236+0.1771j,0.17728+0.16832j,0.1622+0.17247j,0.19334+0.16706j,0.16285+0.23403j,0.10155+0.17965j,0.14809+0.12868j,0.21005+0.15315j,0.19496+0.19726j,0.2261+0.17719j,0.24184+0.26113j,0.21135+0.24818j,0.25789+0.32402j,0.15037+0.37384j,0.13529+0.33385j,0.1048+0.35858j,0.074306+0.32429j,0.074631+0.32767j,0.028734+0.30461j,0.059874+0.24526j,0.091014+0.2728j,0.091339+0.25629j,0.13789+0.29924j,0.076581+0.34044j,0.061499+0.29867j,0.077232+0.3058j,0.062149+0.3154j,0.047067+0.30722j,0.031985+0.28984j,0.047717+0.24604j,0.094264+0.25868j,0.079182+0.26454j,0.12573+0.22809j,0.18768+0.28836j,0.15719+0.3272j,0.20374+0.32654j,0.20407+0.43928j,0.12735+0.45292j,0.11227+0.47267j,0.050968+0.47018j,0.035885+0.44782j,0.020803+0.41698j,0.051943+0.4138j,0.036861+0.418j,0.068+0.41192j,0.068325+0.45013j,0.053243+0.46545j,0.038161+0.46501j,0.053893+0.47935j,0.023403+0.51812j,-0.022493+0.50232j,-0.006761+0.46386j,0.0089713+0.51679j,-0.06774+0.48918j,-0.0057859+0.44274j,-0.0054608+0.48376j,0.010272+0.47601j,-0.0048107+0.50244j,0.026329+0.47115j,0.057469+0.54343j,0.026979+0.56478j,0.027304+0.60646j,-0.034+0.63246j,-0.06449+0.59467j,-0.03335+0.58899j,-0.06384+0.63457j,-0.10974+0.57198j,-0.032375+0.54077j,-0.016642+0.59054j,-0.00091014+0.57478j,0.061044+0.62797j,0.045962+0.70024j,0.015472+0.7416j,-0.015017+0.7413j,0.00071511+0.7692j,-0.045182+0.78756j,-0.029449+0.77977j,-0.059939+0.78741j,0.0020153+0.74096j,0.048562+0.81102j,0.079702+0.81803j,0.14166+0.90364j,0.14198+0.97065j,0.15771+1.0313j,0.18885+1.0807j,0.21999+1.1734j,0.25113+1.2308j,0.31309+1.3464j,0.35963+1.4292j,0.51403+1.5873j,0.62221+1.8707j,0.68416+2.2505j,0.60745+2.7181j,0.37667+3.2037j,-0.039006+3.6128j,-0.5009+3.8672j,-0.96279+4.0748j,-1.4863+4.2185j,-1.9944+4.2743j,-2.5334+4.3576j,-3.2264+4.3358j,-3.9194+4.0964j,-4.5353+3.7432j,-5.1359+3.2454j,-5.5208+2.5566j,-5.6899+1.9629j,-5.8128+1.4018j,-5.8433+0.91063j,-5.8892+0.44644j,-5.8735-0.0017252j,-5.8732-0.41553j,-5.8728-0.86334j,-5.8109-1.3801j,-5.6257-1.8625j,-5.3942-2.3206j,-5.0396-2.6854j,-4.7465-2.9244j,-4.4534-3.1781j,-4.145-3.351j,-3.8981-3.5015j,-3.6205-3.6837j,-3.312-3.7597j,-3.1114-3.8188j,-2.9107-3.913j,-2.7255-4.04j,-2.4633-4.1655j,-2.2318-4.2519j,-1.9388-4.3633j,-1.6457-4.3697j,-1.4143-4.4106j,-1.0904-4.4814j,-0.7049-4.4196j,-0.39643-4.2738j,-0.11877-4.0861j,0.081847-3.8371j,0.15921-3.5677j,0.12872-3.4233j,0.15986-3.3456j,0.17559-3.3085j,0.28377-3.2767j,0.34572-3.2181j,0.48471-3.1685j,0.54667-3.0514j,0.67025-2.9859j,0.70139-2.7746j,0.65549-2.7421j,0.71745-2.6505j,0.65614-2.5781j,0.67188-2.5948j,0.70302-2.5474j,0.70334-2.5468j,0.76529-2.5384j,0.79643-2.5239j,0.90461-2.5078j,0.95116-2.3893j,0.93607-2.336j,0.9364-2.3173j,0.98295-2.2815j,0.93705-2.2243j,0.95278-2.2823j,0.98392-2.2452j,0.99965-2.2867j,1.0462-2.2652j,1.0773-2.3231j,1.1855-2.2952j,1.2167-2.3031j,1.3402-2.2738j,1.3714-2.2186j,1.4487-2.1841j,1.4645-2.1238j,1.511-2.1114j,1.5422-2.0592j,1.5425-2.0459j,1.5736-2.0991j,1.7126-2.1076j,1.79-2.045j,1.8827-2.0064j,1.9293-1.9382j,2.0375-1.9314j,2.1456-1.7854j,2.1614-1.6938j,2.2079-1.5724j,2.162-1.4853j,2.1778-1.4472j,2.1627-1.3686j,2.1322-1.3661j,2.1787-1.363j,2.1945-1.3133j,2.2102-1.3249j,2.2722-1.3162j,2.3495-1.3134j,2.4577-1.2596j,2.5659-1.1848j,2.6278-1.0058j,2.5665-0.94595j,2.6439-0.90114j,2.6288-0.81488j,2.6908-0.82178j,2.7989-0.72949j,2.8455-0.60472j,2.892-0.54782j,3.031-0.38553j,3.0005-0.19011j,3.0471-0.054548j,2.9704+0.17406j,2.8936+0.24546j,2.8478+0.36948j,2.8019+0.44321j,2.756+0.55527j,2.7255+0.62997j,2.6642+0.75527j,2.6337+0.80048j,2.6032+0.96493j,2.5111+1.0274j,2.496+1.1759j,2.3422+1.2947j,2.2655+1.3609j,2.1426+1.4747j,2.0042+1.5233j,1.8967+1.5802j,1.743+1.6289j,1.62+1.627j,1.4817+1.6295j,1.4204+1.5693j,1.3591+1.6366j,1.2362+1.6155j,1.1749+1.6482j,1.0365+1.645j,0.9444+1.6331j,0.83687+1.5971j,0.74475+1.6028j,0.591+1.5179j,0.56051+1.4424j,0.46839+1.389j,0.4225+1.3142j,0.36119+1.2115j,0.40774+1.1466j,0.36184+1.1292j,0.37758+1.0713j,0.34709+1.0724j,0.332+1.0331j,0.30152+1.0169j,0.28643+1.0069j,0.19431+0.98184j,0.17923+0.89539j,0.13333+0.84397j,0.16447+0.73939j,0.18021+0.72088j,0.21135+0.66712j,0.21167+0.65167j,0.28903+0.59843j,0.28936+0.68762j,0.25887+0.61986j,0.30542+0.66415j,0.24411+0.66309j,0.22903+0.64466j,0.21395+0.60169j,0.21427+0.60308j,0.18378+0.5413j,0.23033+0.5145j,0.23065+0.49276j,0.2772+0.473j,0.30834+0.4866j,0.32407+0.52114j,0.30899+0.53914j,0.29391+0.55592j,0.27883+0.54905j,0.26374+0.58265j,0.18703+0.56237j,0.17195+0.49713j,0.18768+0.44873j,0.21882+0.42427j,0.24996+0.41884j,0.25029+0.42117j,0.29683+0.38591j,0.34338+0.45542j,0.31289+0.49119j,0.26699+0.51737j,0.25191+0.45422j,0.28305+0.5138j,0.19093+0.47543j,0.25289+0.44849j,0.19158+0.45834j,0.23813+0.37542j,0.28468+0.41687j,0.285+0.42159j,0.31614+0.44278j,0.30106+0.4991j,0.27057+0.48803j,0.2863+0.53931j,0.17878+0.55593j,0.16369+0.48351j,0.16402+0.4655j,0.17975+0.45583j,0.13385+0.45503j,0.14959+0.36956j,0.21154+0.37743j,0.22727+0.37989j,0.27382+0.40768j,0.25874+0.46257j,0.24366+0.47346j,0.21317+0.50906j,0.18268+0.47753j,0.19841+0.51732j,0.10629+0.5165j,0.10662+0.46166j,0.076126+0.44472j,0.10727+0.37701j,0.13841+0.40165j,0.13873+0.37469j,0.20068+0.4j,0.1702+0.46761j,0.1243+0.45689j,0.10922+0.4499j,0.094134+0.43075j,0.094459+0.43285j,0.06397+0.42434j,0.048887+0.39569j,0.049212+0.34905j,0.095759+0.32957j,0.11149+0.34404j,0.12722+0.34297j,0.14296+0.36689j,0.12787+0.37552j,0.15901+0.38085j,0.12852+0.44971j,0.082627+0.41482j,0.082952+0.44612j,0.0062409+0.40277j,0.052788+0.36474j,0.022298+0.3751j,0.038031+0.32205j,0.06917+0.33155j,0.069495+0.33491j,0.10064+0.33026j,0.10096+0.40857j,0.0088413+0.37847j,0.070796+0.34441j,0.024899+0.4144j,-0.036405+0.34713j,-0.0052658+0.30447j,-0.0049407+0.30573j,-0.0046157+0.24918j,0.072746+0.25246j,0.057664+0.28916j,0.073396+0.27587j,0.073721+0.29178j,0.10486+0.28999j,0.10519+0.36079j,0.028474+0.37344j,0.013392+0.30925j,0.059939+0.33075j,-0.0013652+0.35653j,-0.016447+0.31287j,-0.03153+0.28811j,-0.00039006+0.25125j,-6.501e-05+0.25354j,0.046482+0.22145j,0.062214+0.28508j,0.031725+0.26593j,0.062864+0.28066j,0.032375+0.30502j,0.017293+0.28778j,0.017618+0.29114j,-0.012872+0.28607j,-0.012547+0.25007j,-0.012222+0.24477j,-0.011897+0.19657j,0.050057+0.17914j,0.081197+0.20475j,0.11234+0.22727j,0.11266+0.28635j,0.082172+0.29699j,0.082497+0.32357j,0.0366+0.33788j,0.021518+0.33593j,-0.024379+0.33186j,-0.039461+0.30353j,-0.06995+0.26422j,-0.023403+0.2221j,-0.038486+0.22977j,0.023469+0.1688j,0.054608+0.26571j,-0.006696+0.23123j,0.055258+0.22531j,0.055583+0.27526j,0.0096864+0.30383j,-0.0053958+0.25436j,0.010337+0.3026j,-0.066375+0.25521j,-0.0044207+0.23308j,-0.050318+0.22663j,0.027044+0.18264j,0.027369+0.23811j,0.058509+0.23543j,0.028019+0.2992j,0.028344+0.27604j,-0.0021453+0.34459j,-0.048042+0.29992j,-0.063124+0.33655j,-0.13984+0.26826j,-0.093289+0.22808j,-0.10837+0.22865j,-0.12345+0.19577j,-0.10772+0.1174j,0.00045507+0.11054j,0.016187+0.17928j,0.016512+0.18389j,0.032245+0.22767j,0.0017553+0.24377j,0.017488+0.27783j,-0.074631+0.32558j,-0.12053+0.2539j,-0.13561+0.2288j,-0.11988+0.16593j,-0.10415+0.17075j,-0.073006+0.12763j,-0.057274+0.18657j,-0.087763+0.12684j,-0.010402+0.14799j,-0.025484+0.17876j,-0.025159+0.22322j,-0.086463+0.21829j,-0.086138+0.22364j,-0.16285+0.21495j,-0.19334+0.14189j,-0.17761+0.050874j,-0.10024+0.019543j,-0.053698+0.0072322j,0.0082562+0.039238j,0.023989+0.056466j,0.055128+0.12065j,0.0092314+0.12065j,0.071186+0.16036j,-0.020933+0.2338j,-0.06683+0.19198j,-0.11273+0.16276j,-0.096994+0.093619j,-0.065855+0.077516j,-0.034715+0.083176j,-0.049797+0.079474j,-0.0032505+0.04934j,0.043296+0.095937j,0.012807+0.16595j,-0.048497+0.1217j,0.013457+0.12759j,-0.047847+0.16516j,-0.047522+0.14655j,-0.12423+0.1449j,-0.1085+0.050539j,-0.092769+0.027172j,-4.7868e-16-0.010798j])
xx=np.array([[0.5888,0.50574,0.36562,0.472,-0.0092529,-0.76385,0.33813,0.56706,0.13394,-0.58606,0.21208,-0.34684,-0.7481,0.047943,0.15349,-0.065834,0.43469,-0.93247,-0.29788,-0.0081064,0.030642,0.01624,0.13615,0.24573,-0.61368,0.04302,0.099045,0.14684,-0.31885,-0.046841]
,[0.4666,0.01473,-0.81242,0.27688,0.14901,0.35119,0.74076,0.73619,-0.14391,-0.18985,-0.28051,0.47142,0.53065,0.75568,0.34811,0.037679,-0.30462,0.053585,0.48459,0.0087147,-0.15926,0.31079,0.094357,-0.093648,-0.3137,0.44848,-0.20645,0.040692,-0.43376,-0.062588]
,[0.24782,0.10897,0.65092,-0.11204,0.07968,0.21525,-0.021094,0.22968,-0.22025,0.29019,-0.59825,-0.54782,0.25389,0.53857,0.035823,0.051346,0.73275,-0.12961,0.37984,0.25813,-0.61118,-0.4351,0.40661,0.62939,0.69599,-0.050605,0.90906,0.40242,-0.11415,-0.4218]
,[-0.060488,-0.69053,-0.70466,0.081634,-0.24487,-0.13538,0.24261,0.26407,-0.12055,0.37389,-0.57651,-0.045711,0.13131,0.52731,0.56669,0.01656,-0.26491,0.0087656,-0.27577,0.21685,0.5073,0.30006,0.4439,-0.178,0.27445,0.066822,0.088257,0.72113,-0.98957,0.13435]])
ps=np.array([2,4])
alpha=0.5
acc = np.array([[3, 3], [7, 7], [8, 8]])
print(error(hf,xx,ps,alpha, acc))