import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Rational.hyp_mdpso import hyp_mdpso
from Rational.norm_sig import norm_sig
from Rational.rait.mt_generate import mt_generate
from Rational.rait.multiply_poles import multiply_poles
from Rational.rait.periodize_poles import periodize_poles


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
#print(baseline)
p, c, m, dbest, l, bl, prd = hyp_mdpso(normsig, ps_file, swarm, alpha, iterno, acc, show)
# Reconstructing the signal
mpoles = periodize_poles(multiply_poles(p, m), 1)
mpoles = np.reshape(mpoles, (1, len(mpoles)))
print(mpoles.shape)
aprx = np.real(mt_generate(M, mpoles, c))

# Displaying statistical information
print('------------------- Rational approximation -------------------')
print(f'The approximation error in the sense of PRD: {prd:.2f}%')
print(f'Best dimension index: {dbest}')
print(f'Number of inverse poles in the best dimension: n={len(p)}')
print(f'Inverse pole configuration in the best dimension: m=({", ".join(map(str, m))})')

# Plotting the original ECG and the reconstructed signal
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
x = np.arange(1, M + 1)
axes[0].set_title('Unit Circle')
t = np.linspace(0,np.pi*2,100)
axes[0].plot(np.cos(t), np.sin(t), linewidth=2)
axes[0].grid(True)
axes[0].axis('square')

axes[1].set_title('Original Signal vs. Approximation')
axes[1].plot(x, signal.flatten(), 'b', label='Original Signal', linewidth=2)
axes[1].plot(x, (aprx + baseline).flatten(), 'r', label='Approximation', linewidth=2)
axes[1].grid(True)
axes[1].axis('square')
axes[1].legend()

ax = plt.gca()
ax.set_xlim([100, 300])
ax.set_ylim([-10, 10])

#plt.ylim(-6,5)
#plt.xlim(50,200)
# Saving the resulting plot to a PNG file
plt.savefig('results/rational_signal_approx.png')
#plt.show()
