import numpy as np
from pySymStat import variance_S2, mean_S2

if __name__ == '__main__':
    N = 10
    vecs = np.random.randn(N, 3)
    vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True)
    f = lambda x: x - x ** 2 / 4
    mean = mean_S2(vecs)
    var1 = variance_S2(vecs)
    var2 = variance_S2(vecs, mean = mean)
    assert np.allclose(f(var2), var1)
    print(vecs)
    print(f"mean of {N} projection directions:\n", mean)
    print(f"variance of {N} projection directions (without assigning mean):\n", var1)
    print(f"variance of {N} projection directions (with assigning mean):\n", var2)
