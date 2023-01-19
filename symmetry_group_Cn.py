__all__ = [
    'get_sym_grp_Cn_elems',
    'get_sym_grp_Cn_irreps'
]

import numpy as np
from math import pi, sin, cos

def get_sym_grp_Cn_elems(n):
    elems = np.empty((n, 4), dtype = np.float64)
    for i in range(n):
        elems[i] = np.array([cos(pi * i / n), 0, 0, sin(pi * i / n)])
    return elems

def get_sym_grp_Cn_irreps(n):
    k = n
    rho = np.empty((k, n), dtype = np.ndarray)

    for i in range(k):
        # rho_i : a -> zeta_n^i
        dtype = np.float64 if i == 0 or (n % 2 == 0 and i == n // 2) else np.complex128
        for j in range(n):
            rho[i, j] = np.empty((1, 1), dtype = dtype)
            if i == 0:
                rho[i, j][0, 0] = 1
            elif n % 2 == 0 and i == n // 2:
                rho[i, j][0, 0] = -1 if j % 2 == 1 else 1
            else:
                theta = 2 * pi * i * j / n
                rho[i, j][0, 0] = cos(theta) + 1j * sin(theta)

    return rho
