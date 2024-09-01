import torch

def rho0(z1, z2):
    return torch.abs(z1 - z2) / torch.abs(1 - torch.conj(z2) * z1)