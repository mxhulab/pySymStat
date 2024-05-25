import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    n = 100

    # generated N projection directions represented by unit vectors
    vecs = np.random.randn(n, 3)
    vecs /= np.linalg.norm(vecs, axis = 1)[:, np.newaxis]
    print(vecs)

    print("variance of 100 projection directions (without assigning mean):\n", pySymStat.averaging_S2.variance_S2(vecs))

    print("variance of 100 projection directions (with assigning mean):\n", pySymStat.averaging_S2.variance_S2(vecs, mean = pySymStat.averaging_S2.mean_S2(vecs)))
