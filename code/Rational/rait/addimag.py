import numpy as np

def addimag(v):
    """
    ADDIMAG - Calculates the imaginary part of v using FFT.

    Usage:
        vi = addimag(v)

    Input parameters:
        v : a vector with real elements

    Output parameters:
        vi : a complex vector with appropriate imaginary part
             (to be in Hardy space)

    Copyright: (C) ELTE IK NumAnal, GPL 1.1 ??
    """

    if len(v.shape) != 1:
        raise ValueError("Wrong vector!")
    if not np.all(np.isreal(v)):
        raise ValueError("The vector is not real!")
    
    vf = np.fft.fft(v)
    vif = mt_arrange(vf)
    vi = np.fft.ifft(vif)
    return vi


def mt_arrange(t):
    """
    Rearrange FFT(v) so that lots of zeros appear on the right side of the FFT.
    """

    mt = t.size
    ta = np.copy(t)
    ta[0] = t[0]
    for i in range(1, mt // 2):
        ta[i] = t[i] + np.conj(t[mt - i])
        ta[mt - i] = 0
    return ta

"""
asd=np.array([1,2,3,4,5])
b=addimag(asd)
print(b)
"""