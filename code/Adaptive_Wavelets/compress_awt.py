from Adaptive_Wavelets.opt_params_wt import opt_params_wt
from scipy.optimize import minimize
import numpy as np
import os

def compress_awt(signal, outfile, acc, blocksize, depth, PRD_limit, show=False):
    """
    Last Modified: August 14, 2024.
    Version 1.0.

    compress_awt - Compress an ECG record of the MIT-BIH database by using wavelet transform. 

    Usage: 
        [prd,bits,ecg,atr]=compress_awt(signal,outfile,acc,blocksize,PRD_limit,show)

    Input parameters:
        signal    : discrete time signal
        outfile   : path of the outputfile.
        acc       : the number of bits used for quantizing the coefficients
        blocksize : number of samples of a segment
        PRD_limit : the algorithm works till the approximation error is less than 'prd_limit'   
        show      : optional logical value to display each step of the optimization process

    Output parameters:
        prd     : approximation error of each beat in terms of PRD=(norm(sig-aprx)/norm(sig-mean(sig)))*100;
        bits    : size of the compressed outputfile in bits.
        ecg     : original ECG signal


    This implementation is based on the following papers:
    [1] L. Brechet, M-F. Lucas, C. Doncarli, D. Farina, 
        Compression of Biomedical Signals with Mother Wavelet Optimization 
        and Best-Basis Wavelet Packet Selection, IEEE Transactions on 
        Biomedical Engineering, vol. 54, no. 12, 2007, pp. 2186-2192. 

    [2] M. Abo-Zahhad, A. F. Al-Ajlouni, S. M. Ahmed, R. J. Schilling, 
        A new algorithm for the compression of ECG signals based on mother 
        wavelet parametrization and best-threshold levels slection,
        Digital Signal Processing, vol. 23, 2013, pp. 1002-1011.


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
    # Step 1: Segmenting the ECG signal.
    N = len(signal) // blocksize
    ecg = signal[:N * blocksize]
    Cths = [None] * N
    prd = np.zeros(N)
    q = np.zeros(N)  # Quantization step size.

    # Step 2: Approximating by using adaptive wavelet decomposition.
    init_theta = np.array([1.35980373244182, -0.78210638474440])  # Initial values of the angles (=db3).
    thetas = np.zeros((N, 2))

    # Setting parameters for the optimization.
    MaxIter = 20
    popsize = 14  # Initial population is set based on [1].
    initpop = np.vstack((np.linspace(-np.pi, np.pi, popsize), np.linspace(-np.pi, np.pi, popsize))).T
    initpop[0, :] = init_theta

    for i in range(N):
        print(f'Processing block {i+1}/{N}')
        segment = ecg[i * blocksize:(i + 1) * blocksize]

        # Optimization by using Nelder-Mead simplex algorithm
        result=minimize(objective_function, init_theta, args=(segment, PRD_limit, depth, acc, show), method='Nelder-Mead')
        thetas[i, :] = result.x

        CR, prd[i], Cths[i], q[i], L = opt_params_wt(segment, PRD_limit, depth, acc, False, thetas[i, :])

    # Rounding the quantization step size 'q'.
    maxq = max(q)
    minq = min(q)
    quantq = quantQ(q, maxq, minq, acc[0] * 2)  # These terms quantized by double precision (2*acc[0]).
    q = dequantQ(quantq, maxq, minq, acc[0] * 2)

    # Quantizing the coefficients.
    numcoeff = len(Cths[0])
    C = np.zeros(N * numcoeff)
    for i in range(N):
        C[i * numcoeff:(i + 1) * numcoeff] = np.round(Cths[i] / q[i])
    Cths = C

    # Step 3: Writing the output file.
    outfname = f'{outfile}.dat'
    with open(outfname, 'wb') as fid:
        # Storing the optimal wavelet parameters as the header of the file.
        fid.write(np.int32(N).tobytes())
        fid.write(np.int32(blocksize).tobytes())
        fid.write(np.uint32(numcoeff).tobytes())

        tq = np.pi / (2 ** (acc[0] - 1) - 1)
        for i in range(len(thetas)):
            thetas[i, :] = np.round(thetas[i, :] / tq)
            fid.write(thetas[i, :].tobytes())

        # Storing the level structure.
        fid.write(np.uint8(depth).tobytes())
        fid.write(np.int32(L).tobytes())

        # Storing the quantization step parameters.
        fid.write(np.float64(maxq).tobytes())
        fid.write(np.float64(minq).tobytes())
        fid.write(np.float32(quantq).tobytes())

        # Step 4: Modified run-length coding of the coefficients.
        bs = acc[0]  # Number of bits used to represent each of the quantized coefficients.
        numzero = 0  # Number of subsequent zero.
        for value in Cths:
            if abs(value) < 1e-6:
                if numzero < 2 ** bs - 1:
                    numzero += 1
                else:
                    write_bit(fid, 0)  # Writing a zero flagbit
                    write_bits(fid, numzero, bs)  # Writing number of subsequent zeros
                    numzero = 1
            else:
                if numzero > 0:
                    write_bit(fid, 0)  # Writing a zero flagbit
                    write_bits(fid, numzero, bs)  # Writing number of subsequent zeros
                    numzero = 0

                write_bit(fid, 1)  # Writing a coefficient flagbit
                write_bits(fid, value, bs)    # Writing the quantized coefficient

    # Step 5: Computing the bits that were used for the compression.
    bits = os.path.getsize(outfname) * 8

    return prd, bits, ecg


# AUXILIARY FUNCTIONS

def objective_function(thetas, segment, PRD_limit, depth, acc, show):
    cr, prd, cths, q, l = opt_params_wt(segment, PRD_limit, depth, acc, show, thetas)
    return cr

def write_bit(fid, bit):
    byte = np.packbits(np.array([bit], dtype=np.uint8))
    fid.write(byte.tobytes())

# Write a value using a specific number of bits to the file.
def write_bits(fid, value, bit_count):
    int_value = int(value)
    if int_value < 0:
        int_value = (1 << bit_count) + int_value
    binary_string = format(int_value, f'0{bit_count}b')
    byte_array = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
    fid.write(np.array(byte_array, dtype=np.uint8).tobytes())

# Quantization of the quantization step size.
def quantQ(data, maxdata, mindata, eps):
    d = data - mindata
    q = (maxdata - mindata) / (2 ** eps - 1)
    if q == 0:
        quantdata = np.ones_like(data)
    else:
        quantdata = np.round(d / q)
    return quantdata

# Reconstruction of the quantization step size.
def dequantQ(data, maxdata, mindata, eps):
    q = (maxdata - mindata) / (2 ** eps - 1)
    dequantdata = data * q + mindata
    return dequantdata
