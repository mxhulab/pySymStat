#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['get_sym_grp_table']

import numpy as np

from .quaternion_alg import *

from .quat_set import find_in_quat_set

def get_sym_grp_table(sym_grp):

    sym_grp_table = np.empty((sym_grp.shape[0], sym_grp.shape[0]), dtype = int)

    for (i_g1, i_g2), _ in np.ndenumerate(sym_grp_table):
        sym_grp_table[i_g1, i_g2] = find_in_quat_set(quat_mult(sym_grp[i_g1], sym_grp[i_g2]), sym_grp)

    return sym_grp_table
