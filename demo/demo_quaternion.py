import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    v = np.random.randn(3)
    v /= np.linalg.norm(v)

    print(r'R_q v == R_{-q} v:',
          np.allclose(pySymStat.quaternion.quat_rotate(q1, v), pySymStat.quaternion.quat_rotate(-q1, v)))
    print(r'R_{conj(q)} R_q v == v:',
          np.allclose(pySymStat.quaternion.quat_rotate(pySymStat.quaternion.quat_conj(q1), pySymStat.quaternion.quat_rotate(q1, v)), v))
    print(r'R_{q1} R_{q2} v == R_{q1 q2}v:',
          np.allclose(pySymStat.quaternion.quat_rotate(q1, pySymStat.quaternion.quat_rotate(q2, v)), pySymStat.quaternion.quat_rotate(pySymStat.quaternion.quat_mult(q1, q2), v)))
