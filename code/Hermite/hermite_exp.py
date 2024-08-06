"""
Computing the Hermite expansion of a heart beat.
hermite_exp - Segmenting ECG signals into heart beats.

Usage: 
    [aprx,bm,la,co,b,prds]=hermite_exp(beat,onsets,offsets,basenums,acc,opt,show)

Input parameters:
    beat     : samples of the heart beat that should be compressed  
    onsets   : onset index of the QRS complex of the heart beat 
    offsets  : offset index of the QRS complex of the heart beat 
    basenums : number of terms in the expansions related to [P,QRS,T] waves.
            For instance basenums=[2,7,6] means that we use 2,7,6 
            coefficients for representing the P,QRS,T wave, respectively.
    acc      : number of bits that is used for quantizing the parameters: coefficients, dilations, translations, etc.
    opt      : parameters for the optimization, which is an iterative process. 
            Namely, we try all the values in the interval [opt.lowerbound,opt.upperbound].
            The distance of each value in this discrete interval is opt.step. 
    show     : displaying the results at each step of the optimization 

Output parameters:
    aprx : aprx(i) contains the approximation of the ith section
    bm   :  The heart beat is segmented into different sections (or waves):
        - 'bm(i,1)' contains the length of the ith section
        - 'bm(i,2)' contains the translation parameter of the Hermite system of ith section
    la   : contains the boundary values of each section
    co   : it is a cell array, co{i} contains the coefficients of the ith section 
    b    : 'b(i)' contains the dilation parameter of the Hermite system of ith section
    prds :  prd(i) contains the error of the approximation of the ith section
            For ECG: prds(1),prds(2),prds(3) corresponds to the prd of the P,QRS,T waves, respectively.

NOTE: WFDB PhysioToolkit package should be loaded in MATLAB!!!

Copyright (c) 2017, Péter Kovács kovika@inf.elte.hu>  
Eötvös Loránd University, Budapest, Hungary, 2017.   

This implementation is based on the following papers:

[1] R. Jane, S. Olmos, P. Laguna, P. Caminal, 
    Adaptive Hermite models for ECG data compression: performance and evaluation with automatic wave detection, 
    Proceedings of Computers in Cardiology, 1993, pp. 389-392. 

[2] A. Sandryhaila, S. Saba, M. Puschel, J. Kovacevic, 
    Efficient Compression of QRS Complexes Using Hermite Expansion, 
    IEEE Transactions on Signal Processing, vol. 60, no. 2, 2012, pp. 947-955. 

[3] T. Dózsa, P. Kovács, 
    ECG signal compression using adaptive Hermite functions, 
    Advances in Intelligent Systems and Computing, vol. 399, 2015, pp. 245-254. 

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from Hermite.hermite_roots import hermite_roots
from Hermite.hermite_coeff import hermite_coeff
from Hermite.hermite_system import hermite_system
from Hermite.hermite_recbeat import hermite_recbeat


def hermite_exp(beat, onsets, offsets, basenums, acc, lowerbound, upperbound, step, show=False):
    QRS_on = onsets
    QRS_off = offsets

    QRS = beat[QRS_on-1:QRS_off ] # QRS complex
    P = beat[:QRS_on ]            # P wave + PT interval
    T = beat[QRS_off-1:]          # T wave

    wp = [0, QRS_on-1, QRS_off-1, len(beat) - 1]
    la = beat[wp]
    
    segments = [P, QRS, T]
    bm = np.zeros((3, 2), dtype=int)
    aprx = [0, 0, 0]
    aprxq = [0, 0, 0]
    best_co = [None, None, None]   # Original coefficients
    best_qco = [None, None, None]  # Quantized coefficients
    best_b = [0, 0, 0]             # Translation is zero for all waves (P, QRS, T)
    best_t = []
    best_err = [float('inf'), float('inf'), float('inf')]
    padding = 0
    hmsx = [None, None, None]

    # Normalizing the baseline in each segment
    segment0 = []
    for i, segment in enumerate(segments):
        m = (segment[-1] - segment[0]) / (len(segment) - 1)
        x = np.arange(len(segment))
        base_line = segment[0] + m * x
        seg0 = segment - base_line
        seg0 = np.concatenate(([0] * padding, seg0, [0] * padding))
        best_t.append(np.ceil(len(seg0) / 2))
        hmsx[i] = np.arange(len(seg0)) - best_t[i] + 1
        bm[i, :] = [len(segment), best_t[i] - padding]
        segment0.append(seg0)

    # Precomputing matrices to speed up the optimization
    rootnum = np.zeros(len(segment0), dtype=int)
    alpha = [None] * len(segment0)
    Lambda = [None] * len(segment0)
    HMS = [None] * len(segment0)
    tk = [None] * len(segment0)
    
    for i, seg0 in enumerate(segment0):
        rootnum[i] = len(seg0)
        alpha[i] = hermite_roots(rootnum[i])
        hms = hermite_system(alpha[i], len(alpha[i]))
        HMS[i] = hms[:, :basenums[i]] 
        Lambda[i] = np.matmul(hms, hms.T) 
        M = len(seg0)
        if M % 2 == 0:
            tk[i] = np.arange(-M//2, M//2)
        else:
            tk[i] = np.arange(-np.floor(M/2), np.floor(M/2) + 1)
    
    for b in np.arange(lowerbound, upperbound, step):    
        for i, seg in enumerate(segments):
            # Computing the coefficients of the expansion
            co = hermite_coeff(segment0[i], 1 / b, 0, alpha[i], Lambda[i], HMS[i])
            # Reconstructing the signal by using thresholded coefficients            
            uniform_hms = hermite_system(tk[i] / b, basenums[i])
            qco = quant(co, acc)
            rec_segq = (uniform_hms @ qco).T  # Reconstruction using uniform sampling without quantization
            rec_seg = (uniform_hms @ co).T    # Reconstruction using uniform sampling with quantized coefficients
            
            err = round(np.sqrt(np.sum((segment0[i] - rec_segq) ** 2) / np.sum((segment0[i] - np.mean(segment0[i])) ** 2)) * 100)
            
            if best_err[i] > err:
                best_err[i] = err
                best_co[i] = co
                best_qco[i] = qco
                best_b[i] = b
                hms = hermite_system(hmsx[i], basenums[i], b)
                aprx[i] = rec_seg
                aprxq[i] = rec_segq
        
        # Displaying the approximation at each step
        if show:
            best_co_sparse = [csr_matrix(c).T for c in best_co]
            best_qco_sparse = [csr_matrix(c).T for c in best_qco]
            tk_sparse = [csr_matrix(t).T for t in tk]

            beat_aprx = hermite_recbeat(bm, la, best_co_sparse, best_b, tk_sparse)
            beat_aprxq = hermite_recbeat(bm, la, best_qco_sparse, best_b, tk_sparse)
            plt.cla()
            plt.plot(beat, 'b', linewidth=2)
            plt.plot(beat_aprxq, 'r', linewidth=1)
            prd = np.sqrt(np.sum((beat - beat_aprx) ** 2) / np.sum((beat - np.mean(beat)) ** 2)) * 100
            prdq = np.sqrt(np.sum((beat - beat_aprxq) ** 2) / np.sum((beat - np.mean(beat)) ** 2)) * 100
            legend = ['Original signal', f'Apprx. (PRD: {prdq:.2f} %)']
            plt.legend(legend)
            plt.axis([0, len(beat), np.min(beat) - 0.05, np.max(beat) + 0.05])
            plt.draw()
            plt.pause(0.05)
    
    for i in range(len(segments)):
        if padding>0:
            aprx[i] = aprx[i][padding:-padding]
        m = (la[i + 1] - la[i]) / (len(aprx[i]) - 1)
        x = np.arange(len(aprx[i]))
        base_line = la[i] + m * x
        aprx[i] += base_line

    # Displaying the final result
    if show:
        plt.figure(1)
        plt.plot(beat)
        plt.plot(np.arange(QRS_on-1, QRS_off ), QRS, 'r', linewidth=2)
        plt.plot(np.arange(QRS_on-1), P[:-1], 'g', linewidth=2)
        plt.plot(np.arange(QRS_off , len(beat) ), T[1:], 'k', linewidth=2)
        plt.plot(wp, la, 'r.')
        plt.plot([QRS_on, QRS_off], beat[[QRS_on, QRS_off]], 'g.')
        plt.show()
    """
        tt = np.round([(QRS_on + 1) / 2, (QRS_on + QRS_off) / 2, (QRS_off + len(beat)) / 2]).astype(int)
        beat_aprx = np.concatenate([aprx[0][:-1], aprx[1][:-1], aprx[2]])
        plt.plot(beat, 'b', linewidth=2)
        plt.plot(beat_aprx, 'r', linewidth=2)
        prd = np.sqrt(np.sum((beat - beat_aprx) ** 2) / np.sum((beat - np.mean(beat)) ** 2)) * 100
        plt.plot([tt[0], tt[0]], [0, beat[tt[0]]], 'k-', linewidth=2)
        plt.plot([tt[1], tt[1]], [0, beat[tt[1]]], 'k-', linewidth=2)
        plt.plot([tt[2], tt[2]], [0, beat[tt[2]]], 'k-', linewidth=2)
        plt.plot(beat, 'b', linewidth=2)
        plt.plot(beat_aprx, 'r', linewidth=2)
        plt.plot([0, QRS_on, QRS_off, len(beat) - 1], beat[[0, QRS_on, QRS_off, -1]], 'g.', markersize=18)
        plt.legend(['Original signal', f'Apprx. (PRD: {prd:.2f} %)'], fontsize=14)
        plt.axis('tight')
        plt.grid(True)
        plt.show()
    """

    co = best_qco
    prds = np.zeros(len(co))
    for i in range(len(co)):
        co[i] = co[i].T
        # Computing the prd of each segment
        prds[i] = np.linalg.norm(segments[i] - aprx[i]) / np.linalg.norm(segments[i] - np.mean(segments[i])) * 100

    b = best_b
    best_qco_sparse = [csr_matrix(c).T for c in best_qco]
    tk_sparse = [csr_matrix(t).T for t in tk]
    aprx = hermite_recbeat(bm, la, best_qco_sparse, b, tk_sparse)

    return aprx, bm, la, co, b, prds

# AUXILIARY FUNCTIONS 

def quant(data, eps):
    qr=2/(2**eps-1)
    return np.round(data/qr)*qr