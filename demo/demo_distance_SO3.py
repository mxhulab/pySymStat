import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)

    print("spatial rotation q1: ", q1)

    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    print("spatial rotation q2: ", q2)

    print("arithmetic distance between q1 and q2: \t", pySymStat.distance.distance_SO3(q1, q2, 'arithmetic'))
    print("geometric distance between q1 and q2: \t" , pySymStat.distance.distance_SO3(q1, q2, 'geometric'))
