import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Rational.hyp_mdpso import hyp_mdpso
from Rational.norm_sig import norm_sig
from Rational.rait.mt_generate import mt_generate
from Rational.rait.multiply_poles import multiply_poles
from Rational.rait.periodize_poles import periodize_poles
from Rational.Hyperbolic_operators.draw_unitcircle import draw_unitcircle


# ------------- Signal approximation by using orthogonal rational functions -------------

# Setting parameters.
ecg_dbdir = "data/"
rec_name = "PhysioNet_MITBIH_rec119_5min.mat"
ps_file = "data/poleconf.mat"
acc = np.array([[3, 3], [7, 7], [8, 8]])
alpha = 0.5
iterno = 20
swarm = 30

# Loading an ECG record
mat_contents = sio.loadmat(ecg_dbdir + rec_name)
beats = mat_contents["beats"]
ecg = mat_contents["ecg"]

show = False  # <- Do not turn on for this example program.
k = 20 # <- Number of beats to approximate

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].set_facecolor("lightblue")
axes[1].set_facecolor("lightblue")

for i in range(k):
    
    # Adaptive representation of the kth heartbeat in an ECG record
    signal = beats[i]
    signal = signal[0]
    M = len(signal)
    signal = np.reshape(signal, (1, M))  # signal should be a row vector.
    signal = signal[0]
    normsig, baseline = norm_sig(signal)  # normalizing the baseline of the signal.
    p, c, m, dbest, l, bl, prd = hyp_mdpso(
        normsig, ps_file, swarm, alpha, iterno, acc, show
    )

    # Reconstructing the signal
    mpoles = periodize_poles(multiply_poles(p, m), 1)
    mpoles = np.reshape(mpoles, (1, len(mpoles)))
    aprx = np.real(mt_generate(M, mpoles, c))

    # Plotting the original ECG and the reconstructed signal    
    axes[1].cla()
    draw_unitcircle(p, m)
    unitcircle_title=axes[1].get_title()

    signal=signal.flatten()
    aprx=(aprx + baseline).flatten()
    delay=60 # <- Approximation delay on the X axis

    range_=range(0,M+delay,7)
    range_=list(range_) +[M+delay]
    for j in range_:
        axes[0].cla()
        axes[0].set_title(f"Original Signal vs. Approximation ({i+1}/{k})\nError in the sense of PRD: {prd:.2f}%")
        axes[1].set_title(unitcircle_title + f"\nBest dimension index: {dbest}")
        if j<M:
            x = np.arange(0, j)
            axes[0].plot(x, signal[0:j], "b", label="Original Signal", linewidth=4)
        else:
            x = np.arange(0, M)
            axes[0].plot(x, signal, "b", label="Original Signal", linewidth=4)

        if j>delay:
            axes[0].plot(x[0:j-delay], aprx[0:j-delay], "r", label="Approximation", linewidth=2)
        
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_ylim([-6, 6])
        axes[0].set_xlim([0, M])
        plt.pause(0.01)
        
plt.show()

