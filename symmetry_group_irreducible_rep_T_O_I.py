#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['get_sym_grp_irreducible_rep_T', \
           'get_sym_grp_irreducible_rep_O', \
           'get_sym_grp_irreducible_rep_I']

from math import pi, cos, sin, sqrt
from functools import reduce
import numpy as np

from .symmetry_group_elems_T_O_I import get_sym_grp_elems_T, get_sym_grp_elems_O, get_sym_grp_elems_I

from .quaternion_alg import *

def convert_sym_grp_ab_from_string_to_matrix(s, a, b):
    m = map(lambda c : a if c == 'a' else b, s)
    return reduce(lambda x, y : x @ y, m, np.eye(a.shape[0], dtype = a.dtype))

def orthonormalize(rho):
    # orthonormalization of a real irrep.

    N = rho.shape[0]
    d = rho[0].shape[0]
    A = np.zeros((d, d), dtype = np.float64)

    for (i, j), _ in np.ndenumerate(A):
        A[i, j] = sum([np.vdot(rho[g][:, i], rho[g][:, j]) for g in range(N)]) / N

    V = np.linalg.cholesky(A).T
    Vinv = np.linalg.inv(V)

    # rho = map(lambda x : V @ x @ Vinv, rho)
    for i in range(N):
        rho[i] = V @ rho[i] @ Vinv

def get_sym_grp_irreducible_rep_T(sym_grp_ab):

    # print(sym_grp_ab)

    N = 12
    K = 4
    rho = np.empty((K, N), dtype = np.ndarray)

    # a = (1, 2, 3)(4)
    # b = (1, 2)(3, 4)
    # rho_0 : a -> 1, b -> 1
    # rho_1 : a -> zeta_3, b -> 1
    # rho_2 : a -> zeta_3^2, b -> 1
    # rho_3 : a -> [-1 -1 -1], b -> [-1 -1 -1]
    #              [ 1  0  0],      [ 0  0  1]
    #              [ 0  0  1],      [ 0  1  0]
    ab = { 0 : (np.array([[1]], dtype = np.float64), \
                np.array([[1]], dtype = np.float64)), \
           1 : (np.array([[cos(2 * pi / 3) + sin(2 * pi / 3) * 1j]], dtype = np.complex128), \
                np.array([[              1 +                   0j]], dtype = np.complex128)),
           2 : (np.array([[cos(4 * pi / 3) + sin(4 * pi / 3) * 1j]], dtype = np.complex128), \
                np.array([[              1 +                   0j]], dtype = np.complex128)), \
           3 : (np.array([[-1, -1, -1], [1, 0, 0], [0, 0, 1]], dtype = np.float64), \
                np.array([[-1, -1, -1], [0, 0, 1], [0, 1, 0]], dtype = np.float64)) }

    for k in range(K):

        a, b = ab[k]
        for (i, rho_kg) in enumerate(map(lambda x : convert_sym_grp_ab_from_string_to_matrix(x, a, b), sym_grp_ab)):
            rho[k][i] = rho_kg

    orthonormalize(rho[3])

    return rho

def get_sym_grp_irreducible_rep_O(sym_grp_ab):

    # print(sym_grp_ab)

    N = 24
    K = 5
    rho = np.empty((K, N), dtype = np.ndarray)

    # a = (1, 2, 3, 4)
    # b = (1, 3, 2)(4)
    # rho_0 : a -> 1, b -> 1
    # rho_1 : a -> -1, b -> 1
    # rho_2 : a -> [ 1  0], b -> [ 0  1]
    #              [-1 -1],      [-1 -1]
    # rho_3 : a -> [-1 -1 -1], b -> [ 0  1  0]
    #              [ 1  0  0],      [-1 -1 -1]
    #              [ 0  1  0],      [ 0  0  1]
    # rho_4 : a -> [ 1  1  1], b -> [ 0  1  0]
    #              [-1  0  0],      [-1 -1 -1]
    #              [ 0 -1  0],      [ 0  0  1]
    ab = { 0 : (np.array([[1]], dtype = np.float64), \
                np.array([[1]], dtype = np.float64)), \
           1 : (np.array([[-1]], dtype = np.float64), \
                np.array([[ 1]], dtype = np.float64)), \
           2 : (np.array([[1, 0], [-1, -1]], dtype = np.float64), \
                np.array([[0, 1], [-1, -1]], dtype = np.float64)), \
           3 : (np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0]], dtype = np.float64), \
                np.array([[0, 1, 0], [-1, -1, -1], [0, 0, 1]], dtype = np.float64)), \
           4 : (np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype = np.float64), \
                np.array([[0, 1, 0], [-1, -1, -1], [0, 0, 1]], dtype = np.float64)) }

    for k in range(K):

        a, b = ab[k]
        for (i, rho_kg) in enumerate(map(lambda x : convert_sym_grp_ab_from_string_to_matrix(x, a, b), sym_grp_ab)):
            rho[k][i] = rho_kg

    for k in range(2, 5):
        orthonormalize(rho[k])

    return rho

def get_sym_grp_irreducible_rep_I(sym_grp_ab):

    N = 60
    K = 5
    rho = np.empty((K, N), dtype = np.ndarray)

    # a = (1,2)(3,4)(5)
    # b = (2,3,5)
    # rho_0 : a -> 1, b -> 1
    # rho_1 : a -> [-1 -1 -1 -1], b -> [0 0 0 1]
    #              [ 0  0  1  0]       [1 0 0 0]
    #              [ 0  1  0  0]       [0 0 1 0]
    #              [ 0  0  0  1]       [0 1 0 0]
    # rho_2 : a -> [ 0  1  0  0  0], b -> [ 0  0  1  0  0]
    #              [ 1  0  0  0  0]       [ 0 -1 -1 -1 -1]
    #              [ 0  0  1  0  0]       [-1  0  0  0  1]
    #              [-1 -1  0 -1  0]       [ 0  1  0  0  0]
    #              [ 0  1  0  1  1]       [ 1  0  1  0  0]
    # rho_3 : a -> [-x 1 -x], b -> [ 0 1+x -1-x]
    #              [ x x -1]       [-1  -1    x]
    #              [ 0 0 -1]       [-x  -x    1]
    x_rho_3 = ( sqrt(5) - 1) / 2
    x_rho_4 = (-sqrt(5) - 1) / 2
    ab = { 0 : (np.array([[1]], dtype = np.float64), \
                np.array([[1]], dtype = np.float64)), \
           1 : (np.array([[-1, -1, -1, -1], \
                          [ 0,  0,  1,  0], \
                          [ 0,  1,  0,  0], \
                          [ 0,  0,  0, 1]], dtype = np.float64), \
                np.array([[ 0,  0,  0, 1], \
                          [ 1,  0,  0, 0], \
                          [ 0,  0,  1, 0], \
                          [ 0,  1,  0, 0]], dtype = np.float64)), \
           2 : (np.array([[ 0,  1,  0,  0, 0], \
                          [ 1,  0,  0,  0, 0], \
                          [ 0,  0,  1,  0, 0], \
                          [-1, -1,  0, -1, 0], \
                          [ 0,  1,  0,  1, 1]], dtype = np.float64), \
                np.array([[ 0,  0,  1,  0,  0], \
                          [ 0, -1, -1, -1, -1], \
                          [-1,  0,  0,  0,  1], \
                          [ 0,  1,  0,  0,  0], \
                          [ 1,  0,  1,  0,  0]], dtype = np.float64)), \
           3 : (np.array([[-x_rho_3,       1, -x_rho_3], \
                          [ x_rho_3, x_rho_3,       -1], \
                          [       0,       0,       -1]], dtype = np.float64), \
                np.array([[       0, 1 + x_rho_3, -1 - x_rho_3], \
                          [      -1,          -1,      x_rho_3], \
                          [-x_rho_3,    -x_rho_3,            1]], dtype = np.float64)), \
           4 : (np.array([[-x_rho_4,       1, -x_rho_4], \
                          [ x_rho_4, x_rho_4,       -1], \
                          [       0,       0,       -1]], dtype = np.float64), \
                np.array([[       0, 1 + x_rho_4, -1 - x_rho_4], \
                          [      -1,          -1,      x_rho_4], \
                          [-x_rho_4,    -x_rho_4,            1]], dtype = np.float64)) }

    for k in range(K):

        a, b = ab[k]
        for (i, rho_kg) in enumerate(map(lambda x : convert_sym_grp_ab_from_string_to_matrix(x, a, b), sym_grp_ab)):
            rho[k][i] = rho_kg

    for k in range(1, 5):
        orthonormalize(rho[k])

    return rho

if __name__ == '__main__':

    _, sym_grp_ab = get_sym_grp_elems_T()

    print(get_sym_grp_irreducible_rep_T(sym_grp_ab))

    _, sym_grp_ab = get_sym_grp_elems_O()

    print(get_sym_grp_irreducible_rep_O(sym_grp_ab))

    _, sym_grp_ab = get_sym_grp_elems_I()

    print(get_sym_grp_irreducible_rep_I(sym_grp_ab))
