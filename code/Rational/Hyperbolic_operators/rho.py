from Rational.Hyperbolic_operators.rho0 import rho0
import numpy

def custom_atanh(x):
    return 0.5 * numpy.log((1 + x) / (1 - x))

def rho(z1, z2):
    return custom_atanh(rho0(z1, z2))
