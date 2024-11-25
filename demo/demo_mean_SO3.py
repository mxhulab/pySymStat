import numpy as np
from pySymStat import mean_SO3

if __name__ == '__main__':
    N = 10
    quats = np.random.randn(N, 4)
    quats /= np.linalg.norm(quats, axis = 1, keepdims = True)
    print(quats)
    print(f"Mean of {N} spatial rotations:\n", mean_SO3(quats))
