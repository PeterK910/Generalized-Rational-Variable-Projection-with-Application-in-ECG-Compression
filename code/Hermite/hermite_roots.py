import numpy as np


def hermite_roots(N):
    """
    Compute the roots of the Nth Hermite polynomial.

    Parameters:
    - N  : index of the corresponding Hermite polynomial

    Returns:
    - alpha : roots of the Nth Hermite polynomial

    Implemented by Peter Kovacs,
    Department of Numerical Analysis,
    E�tv�s Lorand University, Budapest, Hungary, 2015.

    This implementation is based on the following papers:
    [1] W. Gautschi, Orthogonal polynomials (in Matlab)
        Journal of Computational and Applied Mathematics, vol. 178, 2005, pp. 215-234.
    """

    if N <= 0:
        raise ValueError('N must be greater than zero!')
    
    # Constructing the Jacobian matrix whose eigenvalues are
    # equal to the roots of the Nth Hermite polynomial.
    b = np.sqrt(0.5 * np.arange(1, N))
    J = np.diag(b, -1) + np.diag(b, 1)
    alpha = np.linalg.eigvalsh(J)
    
    return alpha

