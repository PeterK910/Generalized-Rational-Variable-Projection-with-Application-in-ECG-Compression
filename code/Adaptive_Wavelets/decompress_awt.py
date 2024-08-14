from Adaptive_Wavelets.matlab_wavelet_functions import waverec
from Adaptive_Wavelets.genfilt6 import genfilt6
import numpy as np
import os

def decompress_awt(fname, acc):
    """
    Last Modified: August 14, 2024.
    Version 1.0.

    decompress_awt - Decompress a compressed ECG record of the MIT-BIH database. 

    Usage: 
        [ecg,C,L,thetas]=decompress_awt(fname,acc);

    Input parameters:
        fname : path of the compressed data. 
        acc   : the number of bits used for quantizing the coefficients
    Output parameters:
        ecg     : reconstructed ECG signal.
        C       : C(i,:) contains the coefficients of the ith block.
        L       : number of coefficients at each level.
        thetas  : thetas(i,:) parameters of the best wavelet basis of the ith block.


    Copyright (c) 2017, P�ter Kov�cs <kovika@inf.elte.hu>  
    E�tv�s Lorand University, Budapest, Hungary, 2017.   

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

    # Step 1: Reading the input file.
    with open(f"{fname}.dat", 'rb') as fid:

        # Reading the block size and the number of blocks.
        N = np.frombuffer(fid.read(4), dtype=np.int32)[0]  # 'integer*4' in MATLAB
        blocksize = np.frombuffer(fid.read(4), dtype=np.int32)[0]  # 'integer*4' in MATLAB
        cthslen = np.frombuffer(fid.read(4), dtype=np.uint32)[0]  # 'ubit'
        #print(N, blocksize, cthslen)

        # Reading the optimal wavelet parameters from the header.
        tq = np.pi / (2 ** (acc[0] - 1) - 1)
        thetas = np.frombuffer(fid.read(2 * N * 8), dtype=np.float64).reshape(N, 2) * tq
        #print('out',thetas)
        # Reading the level structure.
        depth = np.frombuffer(fid.read(1), dtype=np.uint8)[0]  # 'integer*1' in MATLAB
        L = np.frombuffer(fid.read(4 * (depth + 2)), dtype=np.int32)  # 'integer*4' in MATLAB
        #print(depth, L)
        # Reading the quantization step parameters.
        maxq = np.frombuffer(fid.read(8), dtype=np.float64)[0]  # 'double' in MATLAB
        minq = np.frombuffer(fid.read(8), dtype=np.float64)[0]  # 'double' in MATLAB
        #quantq = np.frombuffer(fid.read(N * (2 * acc[0] // 8)), dtype=np.uint8)  # 'ubit'
        quantq=np.frombuffer(fid.read(4), dtype=np.float32)
        q = dequantQ(quantq, maxq, minq, 2 * acc[0])
        #print(maxq, minq, quantq, q)
        # Reading the coefficients of each block.
        top = 0
        bs = acc[0]
        Cths = np.zeros(N * cthslen, dtype=np.int32)
        while fid.tell() < os.path.getsize(f"{fname}.dat"):
            flag = read_bit(fid)
            if flag == 1:
                Cths[top] = read_bits(fid, bs)
                #print(Cths[top])
                top += 1
            else:
                numzero = read_bits(fid, bs)
                Cths[top:top + numzero] = 0
                #print(numzero, Cths[top:top + numzero])
                top += numzero

    # Restoring coefficients vector.
    Cths = Cths.reshape(N, cthslen)

    # Step 3: Reconstructing the signal.
    ecg = np.zeros(N * blocksize)
    C = np.zeros_like(Cths)
    for i in range(N):
        Lo_D, Hi_D, Lo_R, Hi_R = genfilt6(thetas[i, 0], thetas[i, 1])
        C[i, :] = Cths[i, :] * q[i]
        ecg[i * blocksize:(i + 1) * blocksize] = waverec(C[i, :], L, Lo_R, Hi_R)

    return ecg, C, L, thetas


# AUXILIARY FUNCTIONS

def dequantQ(data, maxdata, mindata, eps):
    # Reconstruction of the quantization step size.
    q = (maxdata - mindata) / (2 ** eps - 1)
    return data * q + mindata

def read_bit(fid):
    # Read a single bit from the file.
    byte = np.frombuffer(fid.read(1), dtype=np.uint8)
    bit = np.unpackbits(byte)[0]
    return bit

def read_bits(fid, bit_count):
    # Read a specific number of bits from the file.
    byte_count = (bit_count + 7) // 8
    byte_array = np.frombuffer(fid.read(byte_count), dtype=np.uint8)
    binary_string = ''.join([format(byte, '08b') for byte in byte_array])
    binary_string = binary_string[:bit_count]
    value = int(binary_string, 2)
    
    if value >= (1 << (bit_count - 1)):
        value -= (1 << bit_count)

    return value