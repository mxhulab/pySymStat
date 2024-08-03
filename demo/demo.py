import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
from pySymStat import meanvar_SO3_G, meanvar_S2_G

if __name__ == '__main__':

    spatial_rotations = np.random.randn(20, 4)
    spatial_rotations /= np.linalg.norm(spatial_rotations, axis = 1)[:, None]
    mean, var, representatives, solutions = meanvar_SO3_G(spatial_rotations, 'C3', verbosity = 1)
    print(spatial_rotations)
    print(mean)
    print(var)
    print(representatives)
    print(solutions)

    projection_directions = np.random.randn(10, 3)
    projection_directions /= np.linalg.norm(projection_directions, axis = 1)[:, None]
    mean, var, representatives, solutions = meanvar_S2_G(projection_directions, 'T', verbosity = 1)
    print(projection_directions)
    print(mean)
    print(var)
    print(representatives)
    print(solutions)
