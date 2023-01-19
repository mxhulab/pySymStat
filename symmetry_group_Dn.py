__all__ = [
    'get_sym_grp_Dn_elems',
    'get_sym_grp_Dn_irreps'
]

import numpy as np
from math import pi, sin, cos

def get_sym_grp_Dn_elems(n):
    elems = np.empty((2 * n, 4), dtype = np.float64)
    for i in range(n):
        elems[i    ] = np.array([cos(pi * i / n), 0, 0, sin(pi * i / n)])
        elems[i + n] = np.array([0, cos(pi * i / n), sin(pi * i / n), 0])
    return elems

def get_sym_grp_Dn_irreps(n):
    k = 2 + (n - 1) // 2 if n % 2 == 1 else 4 + (n - 2) // 2
    rho = np.empty((k, 2 * n), dtype = np.ndarray)

    # rho_0 : a -> 1, b -> 1
    for i in range(2 * n):
        rho[0, i] = np.array([[1]], dtype = np.float64)

    # rho_1 : a -> 1, b -> -1
    for i in range(n):
        rho[1, i    ] = np.array([[ 1]], dtype = np.float64)
        rho[1, i + n] = np.array([[-1]], dtype = np.float64)

    if n % 2 == 1:
        # rho_{i+1} : a -> [cos(2*pi*i/n) -sin(2*pi*i/n)], b -> [1  0]
        #                  [sin(2*pi*i/n)  cos(2*pi*i/n)]       [0 -1]
        for i in range(1, (n + 1) // 2):
            for j in range(n):
                theta = 2 * pi * i * j / n
                rho[i + 1, j    ] = np.array([[cos(theta), -sin(theta)], [sin(theta),  cos(theta)]], dtype = np.float64)
                rho[i + 1, j + n] = np.array([[cos(theta),  sin(theta)], [sin(theta), -cos(theta)]], dtype = np.float64)

    else:
        # rho_2 : a -> -1, b -> 1
        for i in range(n):
            rho[2, i    ] = np.array([[1 if i % 2 == 0 else -1]], dtype = np.float64)
            rho[2, i + n] = np.array([[1 if i % 2 == 0 else -1]], dtype = np.float64)

        # rho_3 : a -> -1, b -> -1
        for i in range(n):
            rho[3, i    ] = np.array([[1 if i % 2 == 0 else -1]], dtype = np.float64)
            rho[3, i + n] = np.array([[-1 if i % 2 == 0 else 1]], dtype = np.float64)

        # rho_{i+3} : a -> [cos(2*pi*i/n) -sin(2*pi*i/n)], b -> [1  0]
        #                  [sin(2*pi*i/n)  cos(2*pi*i/n)]       [0 -1]
        for i in range(1, n // 2):
            for j in range(n):
                theta = 2 * pi * i * j / n
                rho[i + 3, j    ] = np.array([[cos(theta), -sin(theta)], [sin(theta),  cos(theta)]], dtype = np.float64)
                rho[i + 3, j + n] = np.array([[cos(theta),  sin(theta)], [sin(theta), -cos(theta)]], dtype = np.float64)

    return rho
