import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat
# from pySymStat import mean_variance_SO3_G, meanvar_S2_G

if __name__ == '__main__':

    n = 20

    # generate N spatial rotations represented by unit quaternion
    spatial_rotations = np.random.randn(n, 4)
    spatial_rotations /= np.linalg.norm(spatial_rotations, axis = 1)[:, np.newaxis]
    print(spatial_rotations)

    mean, variance, _, _ = pySymStat.averaging_SO3_G.mean_variance_SO3_G(spatial_rotations, 'C3', verbosity = 1)

    print(mean)
    print(variance)
