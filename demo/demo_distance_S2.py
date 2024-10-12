import numpy as np
from pySymStat import distance_S2

if __name__ == '__main__':
    v1 = np.random.randn(3, 2)
    v1 /= np.linalg.norm(v1, axis = 0, keepdims = True)
    v2 = np.random.randn(3, 2)
    v2 /= np.linalg.norm(v2, axis = 0, keepdims = True)
    print("projection direction v1: ", v1)
    print("projection direction v2: ", v2)
    print("arithmetic distance between v1 and v2: \t", distance_S2(v1, v2, 'arithmetic'))
    print("geometric distance between v1 and v2: \t" , distance_S2(v1, v2, 'geometric'))
    print("pairwise arithmetic distance between v1 and v2: \t", distance_S2(v1, v2, 'arithmetic', True))
    print("pairwise geometric distance between v1 and v2: \t" , distance_S2(v1, v2, 'geometric', True))
