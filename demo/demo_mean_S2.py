import numpy as np
from pySymStat import mean_S2

if __name__ == '__main__':
    N = 10
    vecs = np.random.randn(N, 3)
    vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True)
    print(vecs)
    print(f"mean of {N} projection directions:\n", mean_S2(vecs))
