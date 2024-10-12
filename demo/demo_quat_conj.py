import numpy as np
from pySymStat.quaternion import quat_conj

if __name__ == '__main__':
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    print("A quaternion is: \n", q)
    print("The conjugate of this quaternion is: \n", quat_conj(q))
