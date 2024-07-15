import numpy as np
import matplotlib.pyplot as plt
from Hermite.hermite_roots import hermite_roots
from Hermite.hermite_coeff import hermite_coeff
from Hermite.hermite_system import hermite_system
from Hermite.hermite_recbeat import hermite_recbeat


def hermite_exp(beat, onsets, offsets, basenums, acc, lowerbound, upperbound, step, show=False):
    QRS_on = onsets
    QRS_off = offsets

    QRS = beat[QRS_on:QRS_off ]  # QRS complex
    P = beat[:QRS_on ]           # P wave + PT interval
    T = beat[QRS_off:]              # T wave

    wp = [0, QRS_on, QRS_off, len(beat) - 1]
    la = beat[wp]

    segments = [P, QRS, T]

    bm = np.zeros((3, 2), dtype=int)
    aprx = []
    aprxq = []
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
        best_t.append(len(seg0) // 2)
        hmsx[i] = np.arange(len(seg0)) - best_t[i]
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
        Lambda[i] = hms @ hms.T
        M = len(seg0)
        if M % 2 == 0:
            tk[i] = np.arange(-M//2, M//2)
        else:
            tk[i] = np.arange(-M//2, M//2 + 1)

    for b in np.arange(lowerbound, upperbound + step, step):
        for i, seg in enumerate(segments):
            # Computing the coefficients of the expansion
            co = hermite_coeff(segment0[i], 1 / b, 0, alpha[i], Lambda[i], HMS[i])
            # Reconstructing the signal by using thresholded coefficients
            uniform_hms = hermite_system(tk[i] / b, basenums[i])
            qco = quant(co, acc)
            rec_segq = (uniform_hms @ qco).T  # Reconstruction using uniform sampling without quantization
            rec_seg = (uniform_hms @ co).T    # Reconstruction using uniform sampling with quantized coefficients
            print(segment0[i].shape)
            print(rec_segq.shape)
            #exit(0)
            err = round(np.sqrt(np.sum((segment0[i] - rec_segq) ** 2) / np.sum((segment0[i] - np.mean(segment0[i])) ** 2)) * 100)
            
            if best_err[i] > err:
                best_err[i] = err
                best_co[i] = co
                best_qco[i] = qco
                best_b[i] = b
                hms = hermite_system(hmsx[i], basenums[i], b)
                aprx.append(rec_seg)   # *(bm[i,0]+2*padding)
                aprxq.append(rec_segq) # *(bm[i,0]+2*padding)

        # Displaying the approximation at each step
        if show:
            beat_aprx = hermite_recbeat(bm, la, best_co, best_b, tk)
            beat_aprxq = hermite_recbeat(bm, la, best_qco, best_b, tk)
            plt.plot(beat, 'b', linewidth=2)
            plt.plot(beat_aprxq, 'r', linewidth=1)
            plt.show()
            prd = np.sqrt(np.sum((beat - beat_aprx) ** 2) / np.sum((beat - np.mean(beat)) ** 2)) * 100
            prdq = np.sqrt(np.sum((beat - beat_aprxq) ** 2) / np.sum((beat - np.mean(beat)) ** 2)) * 100
            legend = ['Original signal', f'Apprx. (PRD: {prdq:.2f} %)']
            plt.legend(legend)
            plt.axis([0, len(beat), np.min(beat) - 0.05, np.max(beat) + 0.05])
            plt.draw()

    for i in range(len(segments)):
        aprx[i] = aprx[i][padding:-padding]
        m = (la[i + 1] - la[i]) / (len(aprx[i]) - 1)
        x = np.arange(len(aprx[i]))
        base_line = la[i] + m * x
        aprx[i] += base_line

    # Displaying the final result
    if show:
        plt.figure(1)
        plt.plot(beat)
        plt.plot(np.arange(QRS_on, QRS_off + 1), QRS, 'r', linewidth=2)
        plt.plot(np.arange(1, QRS_on), P[:-1], 'g', linewidth=2)
        plt.plot(np.arange(QRS_off + 1, len(beat) + 1), T[1:], 'k', linewidth=2)
        plt.plot(wp, la, 'r.')
        plt.plot([QRS_on, QRS_off], beat[[QRS_on, QRS_off]], 'g.')
        plt.show()

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

    co = best_qco
    prds = np.zeros(len(co))
    for i in range(len(co)):
        co[i] = co[i].T
        # Computing the prd of each segment
        prds[i] = np.linalg.norm(segments[i] - aprx[i]) / np.linalg.norm(segments[i] - np.mean(segments[i])) * 100

    b = best_b
    aprx = hermite_recbeat(bm, la, best_qco, b, tk)

    return aprx, bm, la, co, b, prds

# AUXILIARY FUNCTIONS 

def quant(data, eps):
    qr=2/(2**eps-1)
    return np.round(data/qr)*qr