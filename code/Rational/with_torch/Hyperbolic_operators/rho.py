from Rational.with_torch.Hyperbolic_operators.rho0 import rho0
import torch

def custom_atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

def rho(z1, z2):
    return custom_atanh(rho0(z1, z2))
