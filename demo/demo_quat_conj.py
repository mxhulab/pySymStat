import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    q = np.random.randn(4)
    q /= np.linalg.norm(q)

    print("A quaternion is: \n", q)

    print("The conjugate of this quaternion is: \n", pySymStat.quaternion.quat_conj(q))
