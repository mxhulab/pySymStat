import numpy as np
from math import *
from numpy.typing import NDArray
from typing import Tuple, List
from .generate import generate_by_two, generate_irreps

qa = np.array([sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2], dtype = np.float64)
qb = np.array([0.5        , 0.5, 0.5, 0.5        ], dtype = np.float64)
qe = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float64)
elems, elems_ab = generate_by_two(qa, qb, [], [], qe, '')
elems = np.vstack(elems)
assert elems.shape == (24, 4)
rhos = generate_irreps([
    (
        np.array([[1]], dtype = np.float64),
        np.array([[1]], dtype = np.float64)
    ),
    (
        np.array([[-1]], dtype = np.float64),
        np.array([[ 1]], dtype = np.float64)
    ),
    (
        np.array([
            [ 1,  0],
            [-1, -1]
        ], dtype = np.float64),
        np.array([
            [ 0,  1],
            [-1, -1]
        ], dtype = np.float64)
    ),
    (
        np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0]
        ], dtype = np.float64),
        np.array([
            [ 0,  1,  0],
            [-1, -1, -1],
            [ 0,  0,  1]
        ], dtype = np.float64)
    ),
    (
        np.array([
            [ 1,  1, 1],
            [-1,  0, 0],
            [ 0, -1, 0]
        ], dtype = np.float64),
        np.array([
            [ 0,  1,  0],
            [-1, -1, -1],
            [ 0,  0,  1]
        ], dtype = np.float64)
    )
], elems_ab)

def get_grp_info(n : int) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[int]]:
    '''Get elements (in unit quaternion) and real irreps of octahedral group O.

    Parameters
    ----------
    n : int
        this is a dummy variable.

    Returns
    -------
    elems : array of shape (24, 4)
    irreps: List of real irreps
        `irreps[k]` is the k-th irrep of shape = (24, dk, dk),
        where dk is the dimension of k-th real irrep.
    s_irreps : List of int
        `s_irreps[k]` is the degree of split of k-th real irrep.
    '''
    return elems, rhos, [1, 1, 1, 1, 1]
