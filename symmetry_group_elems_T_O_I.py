#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['get_sym_grp_elems_T', \
           'get_sym_grp_elems_O', \
           'get_sym_grp_elems_I']

from math import pi, cos, sin, sqrt
import numpy as np

from .quaternion_alg import *

from .quat_set import is_in_quat_set, find_in_quat_set

def put_quat_into_sym_grp(sym_grp, sym_grp_ab, sym_grp_len, q, ab, qa, qb):
    """
    qa, generator a
    qb, generator b
    q, the quaternion to be put
    s, the ab generator string corresponds to q
    sym_grp_len, current length of symmetry grp
    """

    if not is_in_quat_set(q, sym_grp[:sym_grp_len[0]]):

        sym_grp[sym_grp_len[0]] = q
        sym_grp_ab[sym_grp_len[0]] = ab

        sym_grp_len[0] += 1

        # print(sym_grp_len)

        put_quat_into_sym_grp(sym_grp, sym_grp_ab, sym_grp_len, quat_mult(qa, q), 'a' + ab, qa, qb)
        put_quat_into_sym_grp(sym_grp, sym_grp_ab, sym_grp_len, quat_mult(qb, q), 'b' + ab, qa, qb)

def get_sym_grp_elems_T():

    a = np.array([0.5, 0, 0,           sqrt(3) / 2], dtype = np.float64)
    b = np.array([  0, 0, sqrt(6) / 3, sqrt(3) / 3], dtype = np.float64)

    sym_grp = np.empty((12, 4), dtype = np.float64)
    sym_grp_ab = 12 * ['']

    put_quat_into_sym_grp(sym_grp, sym_grp_ab, [0], np.array([1, 0, 0, 0], dtype = np.float64), '', a, b)

    return sym_grp, sym_grp_ab

def get_sym_grp_elems_O():

    a = np.array([sqrt(2) / 2,   0,   0, sqrt(2) / 2], dtype = np.float64)
    b = np.array([        0.5, 0.5, 0.5,         0.5], dtype = np.float64)

    sym_grp = np.empty((24, 4), dtype = np.float64)
    sym_grp_ab = 24 * ['']

    put_quat_into_sym_grp(sym_grp, sym_grp_ab, [0], np.array([1, 0, 0, 0], dtype = np.float64), '', a, b)

    return sym_grp, sym_grp_ab

def get_sym_grp_elems_I():

    a = np.array([0,   0,                 0,                 1], dtype = np.float64)
    b = np.array([0.5, 0, (sqrt(5) - 1) / 4, (sqrt(5) + 1) / 4], dtype = np.float64)

    sym_grp = np.empty((60, 4), dtype = np.float64)
    sym_grp_ab = 60 * ['']

    put_quat_into_sym_grp(sym_grp, sym_grp_ab, [0], np.array([1, 0, 0, 0], dtype = np.float64), '', a, b)

    return sym_grp, sym_grp_ab

if __name__ == '__main__':

    sym_grp, sym_grp_ab = get_sym_grp_elems_T()

    print(sym_grp)
    print(sym_grp_ab)

    sym_grp, sym_grp_ab = get_sym_grp_elems_O()

    print(sym_grp)
    print(sym_grp_ab)

    sym_grp, sym_grp_ab = get_sym_grp_elems_I()

    print(sym_grp)
    print(sym_grp_ab)
