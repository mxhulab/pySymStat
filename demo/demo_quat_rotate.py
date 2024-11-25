import numpy as np
from pySymStat.quaternion import quat_rotate

if __name__ == '__main__':
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    v = np.random.randn(3)
    print("A quaternion (representing a spatial rotation) is: \n", q)
    print("A vector in 3D space is: \n", v)
    print("Rotate this a vector based on this spatial rotation will result: \n", quat_rotate(q, v))
