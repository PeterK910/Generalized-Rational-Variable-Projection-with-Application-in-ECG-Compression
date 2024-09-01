"""
    Last Modified: August 31, 2024.
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
        with application in rational2 identification", Proceedings of the 21st European
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

from Rational.with_torch.norm_sig import norm_sig
from Rational.with_torch.rait.util import addimag
from Rational.with_torch.rait.mt_sys import mt_coeffs
from Rational.with_torch.rait.mt_sys import mt_system
from Rational.with_torch.rait.mt_sys import mt_generate
from Rational.with_torch.rait.simplex import multiply_poles
from Rational.with_torch.rait.simplex import periodize_poles
from Rational.with_torch.Hyperbolic_operators.scale import scale
from Rational.with_torch.Hyperbolic_operators.rho import rho
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def hyp_mdpso(f, ps_name, s, alpha=0.5, iterno=50, eps=None, show=False, insparts=[]):

    if eps is None:
        eps = 8 * torch.ones((2, 2))

    # Loading polespace
    if isinstance(ps_name, list):
        Dmin = 1
        Dmax = len(ps_name)
        ps = ps_name
    else:
        data = sio.loadmat(ps_name)
        ps = data["ps"]
        ps = ps[0]
        Dmin = data["Dmin"]
        Dmin = Dmin[0][0]
        Dmax = data["Dmax"]
        Dmax = Dmax[0][0]

    # Initializing the swarm
    hf = addimag(f)  # Computing the Hilbert-transform of 'f'.

    xd = torch.randint(1, Dmax + 1, (s,))  # Initializing the particle's dimesions.
    vd = torch.randint(1, Dmax + 1, (s,))  # Initializing the particles's dimension velocities.

    xd_ = xd.clone()  # Initializing the personal best dimensions of the particles.
    gbest = torch.ones(Dmax, dtype=torch.int)  # Initializing the gbest particle at each dimenions.
    
    # Sorting user defined inserted particles
    if insparts:
        if len(insparts) > s:
            raise ValueError("Inserted particles must be smaller than swarm size.")
        else:
            insparts = convert2ps(insparts, Dmax)
    else:
        insparts = [None] * Dmax
    
    # Initializing the compression ratio for each dimension
    cr = torch.ones(len(ps))
    for i in range(len(cr)):
        cn = torch.sum(torch.tensor(ps[i]))  # number of complex coefficients
        pn = len(ps[i])
        cr[i] = (2 * (cn + pn) / len(f)) * 100
    
    # Step 1: Initializing positions and velocities
    xx = [None] * Dmax  # Positions of the particles.
    vx = [None] * Dmax  # Velocities of the particles.
    xy = [None] * Dmax  # Personal best positions of the particles.
    xy_g = [None] * Dmax  # Global best position in each dimension.
    pbesterr_a = [None] * Dmax  # Personal best errors of the particles.
    gbesterr_d = [None] * Dmax  # Global best errors in each dimension.

    for d in range(Dmin - 1, Dmax):
        dim = 2 * len(ps[d][0])  # The poles are complex numbers, so the total dimension of the problem is 2*num_poles.
        xx[d] = torch.zeros((dim, s))
        vx[d] = torch.zeros((dim, s))

        # Initilaizing positions
        r = torch.rand(dim // 2, s)
        phi = 2 * torch.pi * torch.rand(dim // 2, s)
        init_pole = r * torch.exp(1j * phi)
        init_vel = r * torch.exp(1j * phi)
        
        xx[d][0::2, :] = torch.real(init_pole)
        xx[d][1::2, :] = torch.imag(init_pole)

        # Initilaizing velocities
        r = torch.rand(dim // 2, s)
        phi = 2 * torch.pi * torch.rand(dim // 2, s)
        init_vel = r * torch.exp(1j * phi)

        vx[d][0::2, :] = torch.real(init_vel)
        vx[d][1::2, :] = torch.imag(init_vel)

        # Inserting user defined particles to the initial swarm.
        if insparts[d] is not None:
            insdim = insparts[d].shape[1]  # Number of inserted particle at dimension 'd'.
            xx[d][:, :insdim] = insparts[d]  # Overwrite the first insdim number of particles at dimension 'd'.

        err = errors_d(hf, xx[d], ps[d][0], alpha, eps)
        xy[d] = xx[d].clone()

        # Computing the gbest and pbest particles for the initial swarm at each dimension.
        gbest[d] = torch.argmin(err)
        xy_g[d] = xx[d][:, gbest[d]].clone()
        pbesterr_a[d] = err
        gbesterr_d[d] = err[gbest[d]]

    err = errors_gd(hf, xy_g, ps, s, alpha, eps)
    dbest = torch.argmin(err)  # Initializing dbest, the optimum dimension.

    # Initializing parameters for MDPSO
    c1 = 1.5
    c2 = 2
    w_high = 0.8
    w_low = 0.2
    w = w_high
    r = torch.rand(2, 1)
    r1 = torch.rand(Dmax, 1)
    r2 = torch.rand(Dmax, 1)
    Vmax = 0.5
    VDmin = 0
    VDmax = 3
    maxPD = 8  # Maximal number of particles in each dimension.

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    maxpn = len(ps[-1][0][0:-1])  # Maximal number of poles in the highest dimension.

    for t in range(1, iterno + 1):
        err = errors_xd(hf, xx, xd, ps, s, alpha, eps)

        for a in range(s):
            if err[a] < pbesterr_a[xd[a] - 1][a]:
                xy[xd[a] - 1][:, a] = xx[xd[a] - 1][:, a].clone()
                pbesterr_a[xd[a] - 1][a] = err[a]

                if err[a] < pbesterr_a[xd_[a] - 1][a]:
                    xd_[a] = xd[a]

            ind = xd == xd[a]
            ind[a] = 0
            err_d = err[ind]
            min_err_d = min(err_d) if len(err_d) > 0 else float("inf")

            if err[a] < min(gbesterr_d[xd[a] - 1], min_err_d):  # (Step 3.1.4.)
                gbest[xd[a] - 1] = a
                xy_g[xd[a] - 1] = xx[xd[a] - 1][:, a].clone()
                gbesterr_d[xd[a] - 1] = err[a]

                if err[a] < gbesterr_d[dbest]:  # (Step 3.1.4.2.)
                    dbest = xd[a] - 1

        for m in range(1, maxpn + 1):
            for a in range(s):
                # Computing the position updates using the hyperbolic metric.
                if m <= len(ps[xd[a] - 1][0]):
                    cy = array2complex(xy[xd[a] - 1][2 * m - 2 : 2 * m, a])
                    cx = array2complex(xx[xd[a] - 1][2 * m - 2 : 2 * m, a])
                    cy_ = array2complex(xy_g[xd[a] - 1][2 * m - 2 : 2 * m])
                    cv = array2complex(vx[xd[a] - 1][2 * m - 2 : 2 * m, a])

                    _, term1 = scale(0, cv, w, 0)
                    _, term2 = scale(0, add(-cx, cy), c1 * r1[xd[a] - 1], 0)
                    _, term3 = scale(0, add(-cx, cy_), c2 * r2[xd[a] - 1], 0)
                    term123 = add(add(term1, term2), term3)
                    if rho(0, term123) > Vmax:
                        _, term123 = scale(0, term123, 0.5 / rho(0, term123), 0)

                    vx[xd[a] - 1][2 * m - 2 : 2 * m, a] = torch.reshape(complex2array(term123), (2,))
                    xx[xd[a] - 1][2 * m - 2 : 2 * m, a] = torch.reshape(complex2array(add(cx, term123)), (2,))

                    # Computing the dimension updates.
                    vd[a] = torch.floor(vd[a] + c1 * r[0] * (xd_[a] - xd[a]) + c2 * r[1] * (dbest + 1 - xd[a]))
                    xd_next = xd[a] + Cvd(vd[a], VDmin, VDmax)
                    PD = sum(xd == xd_next)  # Number of particles in dimension xd_next
                    xd[a] = Cxd(xd[a], xd_next, Dmin, Dmax, PD, maxPD)

            r = torch.rand(2, 1)
            r1 = torch.rand(Dmax, 1)
            r2 = torch.rand(Dmax, 1)
            
        w = (w_high - t * (w_high - w_low) / iterno)  # Linearly decreasing the value of 'w'.

        # Displaying the particles at each step.
        if show:

            db = dbest
            gbest_coords = xy_g[db]
            mult = ps[db][0]
            seg = f  # Segmenting 'f' into smaller partitions.
            len_f = len(f)

            # Subtracting the baseline.
            seg, base_line = norm_sig(seg)
            hseg = addimag(seg)
            hseg = hseg.unsqueeze(0)
            period = 1
            tt = torch.linspace(0, 2 * torch.pi, len_f + 1)[:-1]

            # Displaying the actual dbest dimension and multiplicities and lengths.
            print(f"step {t} -----------------------")
            print(f"   Dbest dimension: {dbest}")
            print(f"   Dbest multiplicities: {mult}")
            print(f"   Dbest length: {len_f}")

            # Initializing sz tensor.
            sz = torch.zeros_like(xx[db])
            top_sz = 1
            for i in range(xx[db].shape[1]):
                if db == xd[i]:
                    sz[:, top_sz-1] = xx[db][:, i]
                    top_sz += 1
            sz2 = sz[:, :top_sz].reshape(-1, 1)
            sz = array2complex(sz2).unsqueeze(0).transpose(0, 1)

            # Calculating and quantizing the poles.
            gbest_complex = array2complex(gbest_coords).unsqueeze(0).transpose(0, 1)
            mpoles = multiply_poles(gbest_complex.squeeze(), torch.tensor(mult, dtype=torch.int64))
            mpoles = periodize_poles(mpoles, period)
            mpoles_r = quant(mpoles, "pole", eps)

            # Calculating the coefficients WITHOUT quantized poles.
            c, _ = mt_coeffs(hseg[0].clone().detach().to(torch.cfloat), mpoles.clone().detach().to(torch.cfloat))

            # Calculating the coefficients WITH quantized poles.
            c_r, _ = mt_coeffs(hseg[0].clone().detach().to(torch.cfloat), mpoles_r.clone().detach().to(torch.cfloat))

            # Quantizing the coefficients.
            c_r = quant(c_r, "coeff", eps)

            # Computing the error in terms of PRD.
            fs_r = mt_generate(len_f, mpoles_r, c_r)
            prd_r = 100 * torch.sqrt(torch.sum((seg - torch.real(fs_r)) ** 2) / torch.sum((seg - seg.mean()) ** 2))
            fs = mt_generate(len_f, mpoles, c)
            prd = 100 * torch.sqrt(torch.sum((seg - torch.real(fs)) ** 2) / torch.sum((seg - seg.mean()) ** 2))

            # Computing the compression ratio (CR).
            cn = np.sum(mult) * period  # number of complex coefficients
            pn = len(mult)
            cr = (2 * (cn + pn) / len_f) * 100

            theta = torch.linspace(0, 2 * torch.pi, 100)
            x = torch.cos(theta)
            y = torch.sin(theta)

            ax[1].clear()
            ax[1].plot(x.numpy(), y.numpy(), 'b', linewidth=4)
            ax[1].plot([-1, 1], [0, 0], 'k')
            ax[1].plot([0, 0], [-1, 1], 'k')
            ax[1].set_xlim(-1.02, 1.02)
            ax[1].set_ylim(-1.02, 1.02)
            ax[1].set_title(f"step: {t}")

            styles = ["bo", "bx", "b+", "bs", "bv", "bp", "b.", "bh"]
            styles_best = ["ro", "rx", "r+", "rs", "rv", "rp", "r.", "rh"]

            for j in range(len(mult)):
                real_parts=torch.real(sz[j::len(mult)])
                imag_parts=torch.imag(sz[j::len(mult)])
                ax[1].plot(real_parts, imag_parts, styles[j % len(styles)])

            # Plotting the global best pole configuration in the dbest dimension.
            # Note: sudden changes on this figure indicate changes in the dbest dimension.
            for i in range(len(mult)):
                start_idx = i * 2
                end_idx = start_idx + 2
                asd = array2complex(gbest_coords[start_idx:end_idx])
                asd = quant(asd, "pole", eps).unsqueeze(0).transpose(0, 1)
                ax[1].plot(
                    asd.real.numpy(),
                    asd.imag.numpy(),
                    styles_best[i % len(styles_best)],
                    markersize=15,
                    linewidth=4,
                )

            # Displaying the rational approximation of the segment.
            ax[0].clear()
            ax[0].plot(tt.numpy(), hf.real.numpy(), "b", linewidth=3)
            ax[0].plot(tt[:len_f], torch.real(fs_r) + base_line, "r", linewidth=2)
            ax[0].legend(["Original signal", f"CR: {len_f / (2 * (cn + pn)):.2f}:1, PRD: {prd_r.item():.3f} %"])
            ax[0].axis("tight")
            ax[0].grid(True)
            plt.pause(0.1)



    # Return the gbest poles and the quantized coefficients of the dbest dimension.
    m = ps[dbest][0]
    poles = array2complex(xy_g[dbest])

    # Return dbest length.
    l = len(f)

    # Return the base_line.
    seg, bl = norm_sig(f[:l])
    hseg = addimag(seg)
    hseg = torch.reshape(hseg, (1, len(hseg)))

    # Quantizing the poles.
    poles = quant(poles, "pole", eps).permute(*torch.arange(quant(poles, "pole", eps).ndim - 1, -1, -1))
    p = poles
    m=torch.tensor(m, dtype=torch.int64)
    mpoles = periodize_poles(multiply_poles(poles, m, allow_unique=True), 1)

    # Calculating the coefficients WITH quantized poles.
    c, _ = mt_coeffs(hseg[0].clone().detach().to(torch.cfloat), mpoles.clone().detach().to(torch.cfloat))
    
    # Quantizing the coefficients.
    c = quant(c, "coeff", eps)

    # Calculating the PRD for 'f'.
    aprx = torch.real(mt_generate(l, mpoles, c))
    prd = 100 * torch.sqrt(torch.sum((seg - aprx) ** 2) / torch.sum((seg - torch.mean(seg)) ** 2))

    return p, c, m, dbest, l, bl, prd



def error(hf, x, ps, alpha, eps):
    period = 1
    f = torch.real(hf)
    if len(x.shape) < 2:
        x = x.view(x.shape[0], 1)

    err = torch.torch.zeros(x.shape[1])
    mult = ps.copy()
    length = len(f)
    seg = f.clone()  # Segmenting 'f' into smaller partitions.

    # Subtracting the baseline
    seg, _ = norm_sig(seg)
    hseg = addimag(seg)

    for i in range(x.shape[1]):
        
        if len(mult.shape)>1:
            mult=mult[0]

        # Quantizing the poles
        poles = periodize_poles(multiply_poles(array2complex(x[:, i]).permute(*torch.arange(x[:, i].ndim - 1, -1, -1)), torch.tensor(mult, dtype=torch.int64)), period)
        poles = quant(poles, "pole", eps)
        poles = poles.view(1, -1)

        # Computing the coefficients
        mts = mt_system(length, poles[0])
        hseg_transposed = hseg.permute(*torch.arange(hseg.ndim - 1, -1, -1)).conj()
        hseg_cloned = hseg_transposed.clone().detach()
        co = (mts @ hseg_cloned.to(torch.complex64) / length).permute(*torch.arange(hseg.ndim - 1, -1, -1)).conj()
        
        # Quantizing the coefficients
        co = quant(co, "coeff", eps)

        # Computing the percentage root mean square difference (PRD)
        aprx = torch.real(torch.matmul(co, mts))
        prd = 100 * torch.sqrt(
            torch.sum((seg - aprx) ** 2)
            / torch.sum((seg - torch.mean(seg)) ** 2)
        )

        # Computing the compression ratio
        cn = np.sum(mult)
        pn = len(mult)
        cr = (2 * (cn + pn) / length) * 100

        err[i] = alpha * prd + (1 - alpha) * cr

    return err

def convert2ps(insparts, Dmax):
    inspos = [None] * Dmax
    for i in range(1, Dmax + 1):
        currentdim = [part for part in insparts if part["dim"] == i]
        if currentdim:
            polenum = currentdim[0]["poles"].shape[1]
            inspoles = torch.hstack([part["poles"] for part in currentdim])
            inspos[i - 1] = torch.torch.zeros((2 * inspoles.shape[0], inspoles.shape[1]))
            inspos[i - 1][0::2, :] = torch.real(inspoles)
            inspos[i - 1][1::2, :] = torch.imag(inspoles)
    return inspos

def errors_xd(hf, xx, xd, ps, s, alpha, eps):
    err = torch.torch.zeros(s)
    for a in range(s):
        err[a] = error(hf, xx[xd[a] - 1][:, a], ps[xd[a] - 1], alpha, eps)
    return err

def errors_d(hf, xx, ps, alpha, eps):
    return error(hf, xx, ps, alpha, eps)

def errors_gd(hf, xy_g, ps, s, alpha, eps):
    err = torch.torch.zeros(len(ps))
    for d in range(len(ps)):
        err[d] = error(hf, xy_g[d], ps[d], alpha, eps)
    return err

def rand_r1r2(ps):
    Dmax = len(ps)
    r1 = [None] * Dmax
    r2 = [None] * Dmax
    for i in range(Dmax):
        d = len(ps[i]) * 2
        r1[i] = torch.rand(d, 1)
        r2[i] = torch.rand(d, 1)
    return r1, r2

def Cvd(vd, VDmin, VDmax):
    return torch.clamp(vd, VDmin, VDmax)

def Cxd(xd_prev, xd_next, Dmin, Dmax, PD, maxPD):
    if PD >= maxPD or xd_next < Dmin or xd_next > Dmax:
        return xd_prev
    else:
        return xd_next

def array2complex(ar):
    if len(ar.shape) == 1:
        ar = ar.view(ar.shape[0], 1)

    z = torch.torch.zeros((ar.shape[0] // 2, ar.shape[1]), dtype=torch.complex128)
    for i in range(ar.shape[1]):
        for j in range(ar.shape[0] // 2):
            z[j, i] = ar[2 * j, i] + 1j * ar[2 * j + 1, i]

    z = z.view(1, -1)
    return z[0]

def complex2array(z):
    ar = torch.torch.zeros((2 * len(z), 1), dtype=torch.float64)
    ar[0::2, 0] = torch.real(z).flatten()
    ar[1::2, 0] = torch.imag(z).flatten()
    return ar

def add(z1, z2):
    return (z1 + z2) / (1 + torch.conj(z1) * z2)

def quant(data, roundmode, eps):
    if roundmode == "pole":
        qr = 1 / (2 ** eps[0, 0] - 1)
        qphi = torch.pi / (2 ** (eps[0, 1] - 1) - 1)
    else:
        qr = 1 / (2 ** eps[1, 0] - 1)
        qphi = torch.pi / (2 ** (eps[1, 1] - 1) - 1)

    # Quantizing the angles
    data_phi = torch.round(torch.angle(data) / qphi) * qphi

    # Quantizing the absolute values
    data_r = torch.round(torch.abs(data) / qr) * qr
    if roundmode == "pole":
        data_r[data_r >= 1] = 1 - 1e-6  # Necessary to satisfy abs(poles) < 1

    return data_r * torch.exp(1j * data_phi)
