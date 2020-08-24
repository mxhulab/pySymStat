__all__ = ['distance_SO3', \
           'distance_SO3_G']

import numpy as np

import sys

from .quaternion_alg import *

def distance_SO3(q1, q2):

    return 2 * np.arccos(np.min([1, np.abs(np.dot(q1, q2))]))

def distance_SO3_G(q1, q2, sym_grp):

    # print(q1)

    # print(q1.shape)
    # print(q2.shape)
    # print(sym_grp.shape)

    q1_grp = np.apply_along_axis(lambda q: quat_mult(q1, q), 1, sym_grp)

    # print q1_grp
    
    q1_grp_q2_distance = np.apply_along_axis(lambda q: distance_SO3(q, q2), 1, q1_grp)

    # print q1_grp_q2_distance

    distance_id = np.argmin(q1_grp_q2_distance)

    return q1_grp_q2_distance[distance_id], q1_grp[distance_id, :], sym_grp[distance_id]
