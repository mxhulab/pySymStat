import numpy as np
from math import *
from numpy.typing import NDArray
from typing import Tuple, List

def _get_grp_elems(n : int) -> NDArray[np.float64]:
    elems = np.empty((n, 4), dtype = np.float64)
    for i in range(n):
        elems[i, 0] = cos(pi * i / n)
        elems[i, 1] = 0.
        elems[i, 2] = 0.
        elems[i, 3] = sin(pi * i / n)
    return elems

def _get_grp_irreps(n : int) -> List[NDArray[np.float64]]:
    g = np.arange(n)
    rhos = []
    s = []

    # rho_0 : a -> 1
    rho_0 = np.ones((n, 1, 1), dtype = np.float64)
    rhos.append(rho_0)
    s.append(1)

    # rho_k : a -> [cos(2*pi*k/n) -sin(2*pi*k/n)]
    #              [sin(2*pi*k/n)  cos(2*pi*k/n)]
    # k = 1..[(n-1)/2]
    for k in range(1, (n + 1) // 2):
        rho_k = np.empty((n, 2, 2), dtype = np.float64)
        theta = 2 * pi * k * g / n
        rho_k[:, 0, 0] =  np.cos(theta)
        rho_k[:, 0, 1] = -np.sin(theta)
        rho_k[:, 1, 0] =  np.sin(theta)
        rho_k[:, 1, 1] =  np.cos(theta)
        rhos.append(rho_k)
        s.append(2)

    if n % 2 == 0:
        # rho_{n/2} : a -> -1.
        rho_n2 = np.empty((n, 1, 1), dtype = np.float64)
        rho_n2[:, 0, 0] = np.where(g % 2 == 0, 1, -1)
        rhos.append(rho_n2)
        s.append(1)

    return rhos, s

def get_grp_info(n : int) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[int]]:
    '''Get elements (in unit quaternion form) and real irreps of cyclic group Cn.

    Parameters
    ----------
    n : int

    Returns
    -------
    elems : array of shape (n, 4)
    irreps: List of real irreps
        `irreps[k]` is the k-th irrep of shape = (n, dk, dk),
        where dk is the dimension of k-th real irrep.
    s_irreps : List of int
        `s_irreps[k]` is the degree of split of k-th real irrep.
    '''
    return _get_grp_elems(n), *_get_grp_irreps(n)
