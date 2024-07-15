import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from B_spline.compress import compress
from B_spline.decompress import decompress

ecg_dbdir = 'data/'
rec_name = 'PhysioNet_MITBIH_rec119_5min.mat'

## Loading an ECG record.
# Example: Loading ECG record (replace with actual loading code)
# load([ecg_dbdir,rec_name],'beats','ecg')

mat_contents = sio.loadmat(ecg_dbdir + rec_name)
beats = mat_contents['beats']

## Setting parameters.%
order = 4
prd_limit = 8.0
acc = [8, 8]
show = True  # Displays each step of the optimization.

## Adaptive representation of a heartbeat in an ECG record. 
signal = beats[0][0]
M = len(signal)
signal = np.reshape(signal, (1, M))  # signal should be a row vector.
#print(signal)
init_knot = np.arange(1, M + 1, 2)  # initial knots
s, knots, coeff, prd = compress(signal, order, prd_limit, init_knot, show)
s, prd = decompress(signal, knots, coeff, order, [signal[0, 0], signal[0, -1]], show)

## Plotting the original ECG and the reconstructed signal.
#fig, axes = plt.subplots(1, 2, figsize=(13, 6))
x = np.arange(1, M + 1)
#plt.subplot(1, 2, 1)
plt.gca().set_aspect('auto', adjustable='box')
plt.plot(x, signal.flatten(), 'b', label='Original Signal')
plt.plot(x, s, 'r', linewidth=2, label='Approximation')
plt.stem(knots, s[knots-1], 'r.')
plt.grid(True)
plt.legend()
plt.ylim(-6, 6)

plt.savefig('results/bspline_signal_approx.png')

## Displaying statistical informations.
print('------------------- B-spline approximation -------------------')
print(f'The approximation error in the sense of PRD: {prd:.2f} %')
print(f'Order of the B-splines: {order}')
print(f'Number of knots: {len(knots)}')

plt.show()