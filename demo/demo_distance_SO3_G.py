import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

# from pySymStat.distance import distance_SO3_G

if __name__ == '__main__':

    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)

    print("spatial rotation q1: ", q1)

    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    print("spatial rotation q2: ", q2)

    distance, q1_g = pySymStat.distance.distance_SO3_G(q1, q2, 'O', 'arithmetic')

    print("arithmetic distance between q1 and q2 with molecular symmetry octahedral: \t", distance)

    print("the closest representative of q1 to q2: \t", q1_g)

