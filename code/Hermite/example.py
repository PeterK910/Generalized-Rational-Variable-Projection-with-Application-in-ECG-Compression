import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Hermite.hermite_exp import hermite_exp

ecg_dbdir = "data/"
rec_name = "PhysioNet_MITBIH_rec119_5min.mat"

## Loading an ECG record.
# Example: Loading ECG record (replace with actual loading code)
# load([ecg_dbdir,rec_name],'beats','ecg')

mat_contents = sio.loadmat(ecg_dbdir + rec_name)
beats = mat_contents["beats"]

# Setting parameters.
acc = np.array([8])  # Use 8 bits for quantizing the compressed data: dilation, translation, coefficients, length, std and endpoints of the signal.
basenums = np.array([2, 7, 6,])  # Number of coefficients used for representing P, QRS, T waves, respectively.
onsets = 112
# Onset of the QRS complex.
offsets = 139
# Offset of the QRS complex.
lowerbound = 0.1
upperbound = 65
step = 45 / 50
show = False

# Adaptive representation of a heartbeat in an ECG record.
signal = beats[0][0]
M = len(signal)
signal = np.reshape(signal, (1, M))
# signal should be a row vector.
aprx, bm, la, co, b, prds = hermite_exp(
    signal[0], onsets, offsets, basenums, acc, lowerbound, upperbound, step, show
)
prd = np.linalg.norm(signal - aprx) / np.linalg.norm(signal - np.mean(signal)) * 100

x = np.arange(M)
plt.figure()
plt.gca().set_aspect('auto', adjustable='box')
plt.plot(x, signal.flatten(), 'b', label='Original Signal')
plt.plot(x, aprx, 'r', linewidth=2, label='Approximation')
plt.grid(True)
plt.legend()

plt.savefig('results/hermite_signal_approx.png')


## Displaying statistical informations.
print('------------------- B-spline approximation -------------------')
print(f'The approximation error in the sense of PRD: {prd:.2f} %')
print(f'Number of segments: {len(basenums)}')
print(f'Number of coefficients: {sum(basenums)}')

plt.show()