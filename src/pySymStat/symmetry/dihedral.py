import numpy as np
from math import *
from numpy.typing import NDArray
from typing import Tuple, List

def _get_grp_elems(n : int) -> NDArray[np.float64]:
    elems = np.empty((2 * n, 4), dtype = np.float64)
    for i in range(n):
        elems[i, 0] = cos(pi * i / n)
        elems[i, 1] = 0.
        elems[i, 2] = 0.
        elems[i, 3] = sin(pi * i / n)
    for i in range(n):
        elems[i + n, 0] = 0.
        elems[i + n, 1] = cos(pi * i / n)
        elems[i + n, 2] = sin(pi * i / n)
        elems[i + n, 3] = 0.
    return elems

def _get_grp_irreps(n : int) -> List[NDArray[np.float64]]:
    N = 2 * n
    g = np.arange(n)
    rhos = []
    s = []

    # rho_0 : a -> 1, b -> 1
    rho_0 = np.ones((N, 1, 1), dtype = np.float64)
    rhos.append(rho_0)
    s.append(1)

    # rho_1 : a -> 1, b -> -1
    rho_1 = np.empty((N, 1, 1), dtype = np.float64)
    rho_1[:n] =  1
    rho_1[n:] = -1
    rhos.append(rho_1)
    s.append(1)

    # rho_{k+1} : a -> [cos(2*pi*k/n) -sin(2*pi*k/n)], b -> [1  0]
    #                  [sin(2*pi*k/n)  cos(2*pi*k/n)]       [0 -1]
    # k = 1..[(n-1)/2]
    for k in range(1, (n + 1) // 2):
        rho_k = np.empty((N, 2, 2), dtype = np.float64)
        theta = 2 * pi * k * g / n
        rho_k[:n, 0, 0] =  np.cos(theta)
        rho_k[:n, 0, 1] = -np.sin(theta)
        rho_k[:n, 1, 0] =  np.sin(theta)
        rho_k[:n, 1, 1] =  np.cos(theta)
        rho_k[n:, 0, 0] =  np.cos(theta)
        rho_k[n:, 0, 1] =  np.sin(theta)
        rho_k[n:, 1, 0] =  np.sin(theta)
        rho_k[n:, 1, 1] = -np.cos(theta)
        rhos.append(rho_k)
        s.append(1)

    if n % 2 == 0:
        # rho_{n/2+1} : a -> -1, b -> 1
        rho_n21 = np.empty((N, 1, 1), dtype = np.float64)
        rho_n21[:n, 0, 0] = np.where(g % 2 == 0, 1, -1)
        rho_n21[n:, 0, 0] = np.where(g % 2 == 0, 1, -1)
        rhos.append(rho_n21)
        s.append(1)

        # rho_{n/2+2} : a -> -1, b -> -1
        rho_n22 = np.empty((N, 1, 1), dtype = np.float64)
        rho_n22[:n, 0, 0] = np.where(g % 2 == 0, 1, -1)
        rho_n22[n:, 0, 0] = np.where(g % 2 == 0, -1, 1)
        rhos.append(rho_n22)
        s.append(1)
    return rhos, s

def get_grp_info(n : int) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[int]]:
    '''Get elements (in unit quaternion form) and real irreps of dihedral group Dn.

    Parameters
    ----------
    n : int

    Returns
    -------
    elems : array of shape (2 * n, 4)
    irreps: List of real irreps
        `irreps[k]` is the k-th irrep of shape = (2 * n, dk, dk),
        where dk is the dimension of k-th real irrep.
    s_irreps : List of int
        `s_irreps[k]` is the degree of split of k-th real irrep.
    '''
    return _get_grp_elems(n), *_get_grp_irreps(n)
