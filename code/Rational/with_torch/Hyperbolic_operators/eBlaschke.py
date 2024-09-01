import torch

def eBlaschke(pole, eps, z):
    if isinstance(pole, int):
        pole=torch.tensor(pole)
    Bz = (z - pole) / (1 - torch.conj(pole.clone().detach()) * z)
    Bz = eps * Bz
    return Bz