__all__ = [
    'meanvar_SO3_G',
    'meanvar_S2_G'
]

import numpy as np
from itertools import combinations
from .symmetry_group import get_sym_grp
from .distance import distance_SO3, distance_S2
from .quaternion import quat_mult, quat_conj, quat_rotate
from .NUG import solve_NUG

def mean_SO3(quats, type = 'arithmetic'):
    '''
    Compute the mean of spatial rotations.
    Only arithmetic mean is implemented since there is no
    known algorithm for geometric case.

    parameters
    ==========
    quats : numpy.ndarray of shape (n, 4).
        Unit quaternion representations of spatial rotations.
    type : 'arithmetic' | 'geometric'.
    '''
    if type == 'arithmetic':
        n = len(quats)
        t = np.mean(quats.reshape((n, 4, 1)) * quats.reshape((n, 1, 4)), axis = 0)
        _, eigvects = np.linalg.eigh(t)
        return eigvects[:, -1].copy()

    elif type == 'geometric':
        return NotImplemented

    else:
        raise ValueError('Invalid argument.')

def variance_SO3(quats, type = 'arithmetic', ref = None):
    n = len(quats)
    if ref is None:
        return sum(distance_SO3(quats[i], quats[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / (n * n)
    else:
        return sum(distance_SO3(quats[i], ref, type = type) ** 2 for i in range(n)) / n

def mean_S2(vecs, type = 'arithmetic'):
    '''
    Compute the mean of projection directions.
    Only arithmetic mean is implemented since there is no
    known algorithm for geometric case.

    parameters
    ==========
    vecs : numpy.ndarray of shape (n, 3).
        Unit vector description for projection directions.
    type : 'arithmetic' | 'geometric'.
    '''
    if type == 'arithmetic':
        mean = np.mean(vecs, axis = 0)
        return mean / np.linalg.norm(mean)
    elif type == 'geometric':
        return NotImplemented
    else:
        raise ValueError('Invalid argument.')

def variance_S2(vecs, type = 'arithmetic', ref = None):
    n = len(vecs)
    if ref is None:
        return sum(distance_S2(vecs[i], vecs[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / (n * n)
    else:
        return sum(distance_S2(vecs[i], ref, type = type) ** 2 for i in range(n)) / n

def meanvar_M_G(data, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs):
    sym_grp_elems, sym_grp_table, sym_grp_irreps = get_sym_grp(sym_grp) if isinstance(sym_grp, str) else sym_grp
    n = len(data)
    N = len(sym_grp_elems)

    # Generate f.
    f = np.empty((n, n, N), dtype = np.float64)
    for i, j, g in np.ndindex(n, n, N):
        f[i, j, g] = distance(grp_action(data[i], sym_grp_elems[g]), data[j]) ** 2

    # Solve nug and get results.
    solutions = solve_NUG(f, sym_grp_table, sym_grp_irreps, **kwargs)
    representatives = np.empty_like(data, dtype = np.float64)
    for i in range(n):
        representatives[i] = grp_action(data[i], sym_grp_elems[solutions[i]])
    mean = mean_func(representatives)
    var = variance_func(representatives)
    return mean, var, representatives, solutions

def meanvar_SO3_G(quats, sym_grp, type = 'arithmetic', **kwargs):
    grp_action = quat_mult
    distance = lambda q1, q2: distance_SO3(q1, q2, type = type)
    mean_func = lambda quats: mean_SO3(quats, type = type)
    variance_func = lambda quats: variance_SO3(quats, type = type)
    return meanvar_M_G(quats, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)

def meanvar_S2_G(vecs, sym_grp, type = 'arithmetic', **kwargs):
    grp_action = lambda v, q: quat_rotate(quat_conj(q), v)
    distance = lambda v1, v2: distance_S2(v1, v2, type = type)
    mean_func = lambda vecs: mean_S2(vecs, type = type)
    variance_func = lambda vecs: variance_S2(vecs, type = type)
    return meanvar_M_G(vecs, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)
