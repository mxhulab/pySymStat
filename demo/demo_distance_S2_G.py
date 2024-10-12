import numpy as np
from pySymStat import distance_S2_G, Symmetry

if __name__ == '__main__':
    v1 = np.random.randn(3, 2)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(3, 2)
    v2 /= np.linalg.norm(v2)
    sym = Symmetry('O')
    distance, g, qg, v2g = distance_S2_G(v1, v2, sym, 'arithmetic')
    print("projection direction v1: ", v1)
    print("projection direction v2: ", v2)
    print("arithmetic distance between v1 and v2 with molecular symmetry octahedral: \t", distance)
    print("the closest representative of v1 to v2: \t", v2g)
