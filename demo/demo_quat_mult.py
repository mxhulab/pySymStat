import numpy as np
from pySymStat.quaternion import quat_mult

if __name__ == '__main__':
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)
    print("A quaternion is: \n", q1)
    print("Another quaternion is: \n", q2)
    print("The product of these two quaternions is: \n", quat_mult(q1, q2))
