import numpy as np
from pySymStat import variance_SO3, mean_SO3

if __name__ == '__main__':
    N = 10
    quats = np.random.randn(N, 4)
    quats /= np.linalg.norm(quats, axis = 1, keepdims = True)
    print(quats)
    print(f"variance of {N} spatial rotations (without assigning mean):\n", variance_SO3(quats))
    print(f"variance of {N} spatial rotations (with assigning mean):\n", variance_SO3(quats, mean = mean_SO3(quats)))
