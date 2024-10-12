import numpy as np
from pySymStat import mean_variance_SO3_G

if __name__ == '__main__':
    N = 20
    quats = np.random.randn(N, 4)
    quats /= np.linalg.norm(quats, axis = 1, keepdims = True)
    mean, variance, _, _, _ = mean_variance_SO3_G(quats, 'C3', verbosity = 1)
    print(quats)
    print("mean :\n", mean)
    print("variance :\n", variance)
