#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['spatial_rotation_mean_SO3']

import numpy as np

from .quaternion_alg import *

def tensor(q1, q2):

    assert(q1.dtype == q2.dtype)

    t = np.empty((q1.shape[0], q2.shape[0]), dtype = q1.dtype)

    for (i, j), _ in np.ndenumerate(t):
        t[i, j] = q1[i] * q2[j]

    return t

def spatial_rotation_mean_SO3(spatial_rotations):

    eig_values, eig_vectors = np.linalg.eig(np.mean(np.apply_along_axis(lambda x : tensor(x, x), -1, spatial_rotations), axis = 0))

    mean = eig_vectors[:, np.argmax(eig_values)]

    return mean / np.linalg.norm(mean)

if __name__ == '__main__':

    spatial_rotations = np.random.randn(100, 4)

    spatial_rotations = np.apply_along_axis(lambda x : x / np.linalg.norm(x), -1, spatial_rotations)

    # print(spatial_rotations)
    # print(np.linalg.norm(spatial_rotations, axis = -1))
    
    print(spatial_rotation_mean_SO3(spatial_rotations))

    # print(spatial_rotations[0])
    # print(spatial_rotations[1])

    # print(tensor(spatial_rotations[0], spatial_rotations[1]))
