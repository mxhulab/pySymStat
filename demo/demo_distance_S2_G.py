import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

# from pySymStat.distance import distance_SO3_G

if __name__ == '__main__':

    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)

    print("projection direction v1: ", v1)

    v2 = np.random.randn(3)
    v2 /= np.linalg.norm(v2)

    print("projection direction v2: ", v2)

    distance, v1_g = pySymStat.distance.distance_S2_G(v1, v2, 'O', 'arithmetic')

    print("arithmetic distance between v1 and v2 with molecular symmetry octahedral: \t", distance)

    print("the closest representative of v1 to v2: \t", v1_g)
