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
from .averaging_SO3 import mean_SO3, variance_SO3

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

def meanvar_S2_G(vecs, sym_grp, type = 'arithmetic', **kwargs):
    grp_action = lambda v, q: quat_rotate(quat_conj(q), v)
    distance = lambda v1, v2: distance_S2(v1, v2, type = type)
    mean_func = lambda vecs: mean_S2(vecs, type = type)
    variance_func = lambda vecs: variance_S2(vecs, type = type)

    return meanvar_M_G(vecs, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)
