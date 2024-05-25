import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    projection_directions = np.random.randn(10, 3)
    projection_directions /= np.linalg.norm(projection_directions, axis = 1)[:, np.newaxis]
    print(projection_directions)

    mean, variance, _, _ = pySymStat.averaging_S2_G.mean_variance_S2_G(projection_directions, 'T', verbosity = 1)

    print(mean)
    print(variance)
