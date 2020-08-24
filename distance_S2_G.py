__all__ = ['distance_S2', \
           'distance_S2_G']

import numpy as np

import sys

from .quaternion_alg import *

def distance_S2(v1, v2):

    return np.arccos(np.max([-1, np.min([1, np.dot(v1, v2)])]))

def distance_S2_G(v1, v2, sym_grp):

    assert(isinstance(v1, np.ndarray))
    assert(isinstance(v2, np.ndarray))
    assert(isinstance(sym_grp, np.ndarray))

    assert(v1.shape == (3, ))
    assert(v2.shape == (3, ))
    assert(sym_grp.shape[1] == 4)

    v1_grp = np.apply_along_axis(lambda q: rotate(quat_conj(q), v1), -1, sym_grp)

    v1_grp_v2_distance = np.apply_along_axis(lambda v: distance_S2(v, v2), -1, v1_grp)

    distance_id = np.argmin(v1_grp_v2_distance)

    return v1_grp_v2_distance[distance_id], v1_grp[distance_id, :], sym_grp[distance_id]
