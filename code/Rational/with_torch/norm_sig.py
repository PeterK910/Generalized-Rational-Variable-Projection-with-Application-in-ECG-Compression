import torch

def norm_sig(signal):
    """
    norm_sig - Normalizing the signal by baseline substraction.

    Usage:
        [normsig,base_line]=norm_sig(signal)

    Input parameters:
        signal    : original signal which is given as a ROW vector

    Output parameters:
        normsig   : normalized signal
        base_line : base_line of the signal
    """
    slope = (signal[-1] - signal[0]) / (len(signal) - 1)
    x = torch.arange(1, len(signal) + 1, dtype=signal.dtype, device=signal.device)
    base_line = signal[0] + slope * (x - 1)
    normsig = signal - base_line
    return normsig, base_line
