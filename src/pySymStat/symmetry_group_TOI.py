__all__ = [
    'get_sym_grp_TOI_elems',
    'get_sym_grp_TOI_irreps'
]

import numpy as np
from math import sqrt, pi, sin, cos
from functools import reduce

from .quaternion import *

def generate_sym_grp(qa, qb, elems, elems_ab, q, ab):
    if not any(quat_same(q, q1) for q1 in elems):
        elems.append(q)
        elems_ab.append(ab)
        generate_sym_grp(qa, qb, elems, elems_ab, quat_mult(q, qa), ab + 'a')
        generate_sym_grp(qa, qb, elems, elems_ab, quat_mult(q, qb), ab + 'b')

def get_sym_grp_TOI_elems(sym):
    if sym == 'T':
        qa = np.array([0.5, 0, 0,           sqrt(3) / 2], dtype = np.float64)
        qb = np.array([  0, 0, sqrt(6) / 3, sqrt(3) / 3], dtype = np.float64)
    elif sym == 'O':
        qa = np.array([sqrt(2) / 2,   0,   0, sqrt(2) / 2], dtype = np.float64)
        qb = np.array([        0.5, 0.5, 0.5,         0.5], dtype = np.float64)
    elif sym == 'I' or sym == 'I2':
        qa = np.array([0,   1,                 0,                 0], dtype = np.float64)
        qb = np.array([0.5, (sqrt(5) + 1) / 4, (sqrt(5) - 1) / 4, 0], dtype = np.float64)
    elif sym == 'I1':
        qa = np.array([0,   0,                 0,                 1], dtype = np.float64)
        qb = np.array([0.5, 0, (sqrt(5) - 1) / 4, (sqrt(5) + 1) / 4], dtype = np.float64)
    elif sym == 'I3':
        qa = np.array([0, -0.5257311143, 0, 0.8506508070], dtype = np.float64)
        qb = np.array([0.5, -0.4253254082243262, 0.30901699437494745, 0.6881909629065621], dtype = np.float64)

    elems = []
    elems_ab = []
    generate_sym_grp(qa, qb, elems, elems_ab, np.array([1, 0, 0, 0], dtype = np.float64), '')
    elems = np.array(elems, dtype = np.float64)
    return elems, elems_ab

def orthonormalize(rho):
    n = rho.shape[0]
    d = rho[0].shape[0]
    A = np.zeros((d, d), dtype = np.float64)
    for i, j in np.ndindex(d, d):
        A[i, j] = sum(np.vdot(rho[g][:, i], rho[g][:, j]) for g in range(n)) / n
    V = np.linalg.cholesky(A).T
    Vinv = np.linalg.inv(V)
    for i in range(n):
        rho[i] = V @ rho[i] @ Vinv

def get_sym_grp_TOI_irreps(sym, elems_ab):
    if sym == 'T':
        k = 4
        n = 12
        rho_ab = [(np.array([[1]], dtype = np.float64),
                   np.array([[1]], dtype = np.float64)),
                  (np.array([[cos(2 * pi / 3) + sin(2 * pi / 3) * 1j]], dtype = np.complex128),
                   np.array([[              1                       ]], dtype = np.complex128)),
                  (np.array([[cos(4 * pi / 3) + sin(4 * pi / 3) * 1j]], dtype = np.complex128),
                   np.array([[              1                       ]], dtype = np.complex128)),
                  (np.array([[-1, -1, -1], [1, 0, 0], [0, 0, 1]], dtype = np.float64),
                   np.array([[-1, -1, -1], [0, 0, 1], [0, 1, 0]], dtype = np.float64))]
    elif sym == 'O':
        k = 5
        n = 24
        rho_ab = [(np.array([[1]], dtype = np.float64),
                   np.array([[1]], dtype = np.float64)),
                  (np.array([[-1]], dtype = np.float64),
                   np.array([[ 1]], dtype = np.float64)),
                  (np.array([[1, 0], [-1, -1]], dtype = np.float64),
                   np.array([[0, 1], [-1, -1]], dtype = np.float64)),
                  (np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0]], dtype = np.float64),
                   np.array([[0, 1, 0], [-1, -1, -1], [0, 0, 1]], dtype = np.float64)),
                  (np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype = np.float64),
                   np.array([[0, 1, 0], [-1, -1, -1], [0, 0, 1]], dtype = np.float64))]
    elif sym[0] == 'I':
        k = 5
        n = 60
        x3 = ( sqrt(5) - 1) / 2
        x4 = (-sqrt(5) - 1) / 2
        rho_ab = [(np.array([[1]], dtype = np.float64),
                   np.array([[1]], dtype = np.float64)),
                  (np.array([[-1, -1, -1, -1],
                             [ 0,  0,  1,  0],
                             [ 0,  1,  0,  0],
                             [ 0,  0,  0,  1]], dtype = np.float64),
                   np.array([[ 0,  0,  0, 1],
                             [ 1,  0,  0, 0],
                             [ 0,  0,  1, 0],
                             [ 0,  1,  0, 0]], dtype = np.float64)),
                  (np.array([[ 0,  1,  0,  0, 0],
                             [ 1,  0,  0,  0, 0],
                             [ 0,  0,  1,  0, 0],
                             [-1, -1,  0, -1, 0],
                             [ 0,  1,  0,  1, 1]], dtype = np.float64),
                   np.array([[ 0,  0,  1,  0,  0],
                             [ 0, -1, -1, -1, -1],
                             [-1,  0,  0,  0,  1],
                             [ 0,  1,  0,  0,  0],
                             [ 1,  0,  1,  0,  0]], dtype = np.float64)),
                  (np.array([[-x3,  1, -x3],
                             [ x3, x3,  -1],
                             [  0,  0,  -1]], dtype = np.float64),
                   np.array([[  0, 1 + x3, -1 - x3],
                             [ -1,     -1,      x3],
                             [-x3,    -x3,       1]], dtype = np.float64)),
                  (np.array([[-x4,  1, -x4],
                             [ x4, x4,  -1],
                             [  0,  0,  -1]], dtype = np.float64),
                    np.array([[  0, 1 + x4, -1 - x4],
                              [ -1,     -1,      x4],
                              [-x4,    -x4,       1]], dtype = np.float64))]

    rho = np.empty((k, n), dtype = np.ndarray)
    for i in range(k):
        a, b = rho_ab[i]
        for j in range(n):
            m = map(lambda ch : a if ch == 'a' else b, elems_ab[j])
            rho[i, j] = reduce(lambda x, y : x @ y, m, np.eye(a.shape[0], dtype = a.dtype))
        if rho[i, 0].shape[0] > 1 and rho[i, 0].dtype == np.float64:
            orthonormalize(rho[i])
    return rho
