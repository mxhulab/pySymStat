import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)

    print("projection direction v1: ", v1)

    v2 = np.random.randn(3)
    v2 /= np.linalg.norm(v2)

    print("projection direction v2: ", v2)

    print("arithmetic distance between v1 and v2: \t", pySymStat.distance.distance_S2(v1, v2, 'arithmetic'))
    print("geometric distance between v1 and v2: \t" , pySymStat.distance.distance_S2(v1, v2, 'geometric'))
