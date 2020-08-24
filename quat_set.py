#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['is_in_quat_set', \
           'find_in_quat_set']

import numpy as np

from .quaternion_alg import *

def is_in_quat_set(q, quat_set):

    if quat_set.shape[0] == 0:
        return False
    else:
        return np.any(np.apply_along_axis(lambda elem_quat_set : quat_same(q, elem_quat_set), -1, quat_set))

def find_in_quat_set(q, quat_set):

    v = np.apply_along_axis(lambda elem_quat_set : quat_same(q, elem_quat_set), -1, quat_set)

    return np.where(v)[0][0]
