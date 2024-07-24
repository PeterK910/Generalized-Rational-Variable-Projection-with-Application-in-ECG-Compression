import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Rational.hyp_mdpso import hyp_mdpso
from Rational.norm_sig import norm_sig
from Rational.rait.mt_generate import mt_generate
from Rational.rait.multiply_poles import multiply_poles
from Rational.rait.periodize_poles import periodize_poles
from Rational.Hyperbolic_operators.draw_unitcircle import draw_unitcircle


#------------- Signal approximation by using orthogonal rational functions -------------

# Setting parameters.
ecg_dbdir = 'data/'
rec_name = 'PhysioNet_MITBIH_rec119_5min.mat'
ps_file = 'data/poleconf.mat'
acc = np.array([[3, 3], [7, 7], [8, 8]])
alpha = 0.5
iterno = 20
swarm = 30

# Loading an ECG record
mat_contents = sio.loadmat(ecg_dbdir + rec_name)
beats = mat_contents['beats']
ecg = mat_contents['ecg']

show = False  # <-Displays an animation about the optimization.

# Adaptive representation of the kth heartbeat in an ECG record
k = 0
signal = beats[k]
signal = signal[0]
M = len(signal)
signal=np.reshape(signal, (1, M)) # signal should be a row vector.
signal=signal[0]
normsig, baseline = norm_sig(signal) # normalizing the baseline of the signal.
p, c, m, dbest, l, bl, prd = hyp_mdpso(normsig, ps_file, swarm, alpha, iterno, acc, show)
# Reconstructing the signal
#p=np.array([0.35714-0.61859j,0.21429-0.37115j,0.28571-0.49487j])
#c=np.array([[-0.34356-1.0226j,0.75614-0.7016j,-0.91971+0.50085j,0.17301-0.0086347j,0.39597+0.44869j,-0.0053002-0.21253j,0.22063-0.30687j,-0.0021594+0.086587j,-0.27377-0.099644j,0.013789+0.01918j,0.027346+0.15509j,0.049094-0.061562j,0.070075+0.010562j,-0.094324-0.039761j,0.0042652+0.015159j,-0.010663+0.037899j]])
#m=np.array([4,6,6])

mpoles = periodize_poles(multiply_poles(p, m), 1)
mpoles = np.reshape(mpoles, (1, len(mpoles)))
aprx = np.real(mt_generate(M, mpoles, c))

# Displaying statistical information
print('------------------- Rational approximation -------------------')
print(f'The approximation error in the sense of PRD: {prd:.2f}%')
print(f'Best dimension index: {dbest}')
print(f'Number of inverse poles in the best dimension: n={len(p)}')
print(f'Inverse pole configuration in the best dimension: m=({", ".join(map(str, m))})')

# Plotting the original ECG and the reconstructed signal
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
draw_unitcircle(p,m)

x = np.arange(0, M)

axes[0].set_title('Original Signal vs. Approximation')
axes[0].plot(x, signal.flatten(), 'b', label='Original Signal', linewidth=2)
axes[0].plot(x, (aprx + baseline).flatten(), 'r', label='Approximation', linewidth=2)
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylim([-7, 7])

# Saving the resulting plot to a PNG file
plt.savefig('results/rational_signal_approx.png')
plt.show()
