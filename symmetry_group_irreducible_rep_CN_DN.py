#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['get_sym_grp_irreducible_rep_CN', \
           'get_sym_grp_irreducible_rep_DN']

from math import pi, cos, sin, sqrt
import numpy as np

from .quaternion_alg import *

def get_sym_grp_irreducible_rep_CN(order):

    N = order
    K = order

    rho = np.empty((K, N), dtype = np.ndarray)

    for (k, n), _ in np.ndenumerate(rho):

        rho[k, n] = np.empty((1, 1), dtype = np.float64 if ((k == 0) or (N % 2 == 0 and k == N // 2)) else np.complex128)

        if k == 0:
            rho[k, n][0, 0] = 1
        elif N % 2 == 0 and k == N // 2:
            rho[k, n][0, 0] = -1 if n % 2 == 1 else 1
        else:
            theta = 2 * pi * k * n / N
            rho[k, n][0, 0] = cos(theta) + sin(theta) * 1j

    return rho

def get_sym_grp_irreducible_rep_DN(order):

    N = 2 * order
    K = 2 + (order - 1) // 2 if order % 2 == 1 else 4 + (order - 2) // 2

    rho = np.empty((K, N), dtype = np.ndarray)

    for i in range(N):
        rho[0, i] = np.ones((1, 1), dtype = np.float64)
    for i in range(order):
        rho[1, i] = np.ones((1, 1), dtype = np.float64)
        rho[1, i + order] = np.full((1, 1), -1, dtype = np.float64)

    if order % 2 == 1:

        for k in range((order - 1) // 2):
            for i in range(order):
                theta = 2 * pi * (k + 1) * i / order
                rho[k + 2, i] = np.array([[cos(theta), -sin(theta)], \
                                          [sin(theta),  cos(theta)]], dtype = np.float64)
                rho[k + 2, i + order] = np.array([[cos(theta),  sin(theta)], \
                                                  [sin(theta), -cos(theta)]], dtype = np.float64)

    else:

        for i in range(order):
            rho[2, i] = np.full((1, 1), 1 if i % 2 == 0 else -1, dtype = np.float64)

        rho[2, order:] =  rho[2, :order]
        rho[3, :order] =  rho[2, :order]
        rho[3, order:] = -rho[2, :order]

        for k in range((order - 2) // 2):
            for i in range(order):
                theta = 2 * pi * (k + 1) * i / order
                rho[k + 4, i] = np.array([[cos(theta), -sin(theta)], \
                                          [sin(theta),  cos(theta)]], dtype = np.float64)
                rho[k + 4, i + order] = np.array([[cos(theta),  sin(theta)], \
                                                  [sin(theta), -cos(theta)]], dtype = np.float64)

    return rho

if __name__ == '__main__':

    print(get_sym_grp_irreducible_rep_CN(5))
    print(get_sym_grp_irreducible_rep_DN(5))
    print(get_sym_grp_irreducible_rep_DN(4))
