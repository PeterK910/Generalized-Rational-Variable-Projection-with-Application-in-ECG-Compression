from Adaptive_Wavelets.compress_awt import compress_awt
from Adaptive_Wavelets.decompress_awt import decompress_awt
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os

# ------------- Signal approximation by using orthogonal rational functions -------------

# Loading an ECG record
ecg_dbdir = "data/"
rec_name = "PhysioNet_MITBIH_rec119_5min.mat"
mat_contents = sio.loadmat(ecg_dbdir + rec_name)
beats = mat_contents["beats"]
ecg = mat_contents["ecg"]

# Setting parameters
outfile = 'results/compressed_signal_awt'
acc = np.array([8])  # Use 8 bits for quantizing the compressed data
PRD_limit = 10  # PRD limit for approximation
blocksize = 1024  # Segment size
depth = 7  # Depth of wavelet tree
show = False  # Control for displaying optimization steps

# Adaptive representation of a heartbeat in an ECG record
k = 0  # Example heartbeat index
signal = ecg[(k)*blocksize:(k+1)*blocksize]
M = len(signal)
signal = np.reshape(signal, (M,))

# Adaptive Wavelet Transform (AWT)
prd, bits, ecgsig = compress_awt(signal, outfile, acc, blocksize, depth, PRD_limit, show)
aprx, C, L, thetas = decompress_awt(outfile, acc)

"""
Optionally delete the compressed signal file
"""
if os.path.exists(outfile + '.dat'):
    os.remove(outfile + '.dat')

# Plotting the original ECG and the reconstructed signal
x = np.arange(M)
plt.figure()
plt.gca().set_aspect("auto", adjustable="box")
plt.plot(x, signal.flatten(), 'b', label='Original Signal', linewidth=2)
plt.plot(x, aprx, 'r', label='Approximation', linewidth=2)
plt.grid(True)
plt.legend()

# Save the resulting plot to a PNG file

plt.savefig('results/adaptive_wavelet_approx.png')

# Displaying statistical information
print('---------------- Adaptive wavelet approximation ----------------')
print(f'The approximation error in the sense of PRD: {prd[0]:.2f} %')
print(f'Depth of the wavelet tree: {depth}')
print(f'Blocksize: {blocksize}')

plt.show()
