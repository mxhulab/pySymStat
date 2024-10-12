import numpy as np
from pySymStat import mean_variance_S2_G

if __name__ == '__main__':
    N = 10
    vecs = np.random.randn(N, 3)
    vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True)
    mean, variance, _, _, _ = mean_variance_S2_G(vecs, 'T', verbosity = 1)
    print(vecs)
    print("mean :\n", mean)
    print("variance : \n", variance)
