from Rational.Hyperbolic_operators.rho0 import rho0
import numpy

def rho(z1, z2):
    return numpy.atanh(rho0(z1, z2))