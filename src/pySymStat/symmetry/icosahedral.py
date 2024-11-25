import numpy as np
from math import *
from numpy.typing import NDArray
from typing import Tuple, List
from .generate import generate_by_two, generate_irreps

qa1 = np.array([0.0, 1.0              , 0.0              , 0.0], dtype = np.float64)
qb1 = np.array([0.5, (sqrt(5) + 1) / 4, (sqrt(5) - 1) / 4, 0.0], dtype = np.float64)
qa2 = np.array([0.0, 0.0, 0.0              , 1.0              ], dtype = np.float64)
qb2 = np.array([0.5, 0.0, (sqrt(5) - 1) / 4, (sqrt(5) + 1) / 4], dtype = np.float64)
qa3 = np.array([0.0, -0.5257311143328698, 0.0               , 0.8506508069838757], dtype = np.float64)
qb3 = np.array([0.5, -0.4253254059669702, 0.3090169956761489, 0.6881909591287187], dtype = np.float64)
qe = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float64)
elems1, elems1_ab = generate_by_two(qa1, qb1, [], [], qe, '')
elems1 = np.vstack(elems1)
elems2, elems2_ab = generate_by_two(qa2, qb2, [], [], qe, '')
elems2 = np.vstack(elems2)
elems3, elems3_ab = generate_by_two(qa3, qb3, [], [], qe, '')
elems3 = np.vstack(elems3)
assert elems1.shape == elems2.shape == elems3.shape == (60, 4)
assert elems1_ab == elems2_ab == elems3_ab

x3 = ( sqrt(5) - 1) / 2
x4 = (-sqrt(5) - 1) / 2
rhos = generate_irreps([
    (
        np.array([[1]], dtype = np.float64),
        np.array([[1]], dtype = np.float64)
    ),
    (
        np.array([
            [-x3,  1, -x3],
            [ x3, x3,  -1],
            [  0,  0,  -1]
        ], dtype = np.float64),
        np.array([
            [  0, 1 + x3, -1 - x3],
            [ -1,     -1,      x3],
            [-x3,    -x3,       1]
        ], dtype = np.float64)
    ),
    (
        np.array([
            [-x4,  1, -x4],
            [ x4, x4,  -1],
            [  0,  0,  -1]
        ], dtype = np.float64),
        np.array([
            [  0, 1 + x4, -1 - x4],
            [ -1,     -1,      x4],
            [-x4,    -x4,       1]
        ], dtype = np.float64)
    ),
    (
        np.array([
            [-1, -1, -1, -1],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ], dtype = np.float64),
        np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype = np.float64)
    ),
    (
        np.array([
            [ 0,  1,  0,  0, 0],
            [ 1,  0,  0,  0, 0],
            [ 0,  0,  1,  0, 0],
            [-1, -1,  0, -1, 0],
            [ 0,  1,  0,  1, 1]
        ], dtype = np.float64),
        np.array([
            [ 0,  0,  1,  0,  0],
            [ 0, -1, -1, -1, -1],
            [-1,  0,  0,  0,  1],
            [ 0,  1,  0,  0,  0],
            [ 1,  0,  1,  0,  0]
        ], dtype = np.float64)
    )
], elems1_ab)

def get_grp_info(n : int) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[int]]:
    '''Get elements (in unit quaternion) of icosahedral group In.

    Parameters
    ----------
    n : int
        the index of different conventions for icosahedral groups.

    Returns
    -------
    elems : array of shape (60, 4)
    irreps: List of real irreps
        `irreps[k]` is the k-th irrep of shape = (60, dk, dk),
        where dk is the dimension of k-th real irrep.
    s_irreps : List of int
        `s_irreps[k]` is the degree of split of k-th real irrep.
    '''
    if n == 4:
        raise NotImplementedError('I4 not implemented.')
    return {1 : elems1, 2 : elems2, 3 : elems3}[n], rhos, [1, 1, 1, 1, 1]
