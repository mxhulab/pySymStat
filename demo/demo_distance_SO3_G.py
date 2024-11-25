import numpy as np
from pySymStat import distance_SO3_G, Symmetry

if __name__ == '__main__':
    q1 = np.random.randn(4, 2)
    q1 /= np.linalg.norm(q1, axis = 0, keepdims = True)
    q2 = np.random.randn(4, 2)
    q2 /= np.linalg.norm(q2, axis = 0, keepdims = True)
    sym = Symmetry('O')
    distance, g, qg, q2g = distance_SO3_G(q1, q2, sym, 'arithmetic')
    print("spatial rotation q1: ", q1)
    print("spatial rotation q2: ", q2)
    print("arithmetic distance between q1 and q2 with molecular symmetry octahedral: \t", distance)
    print("the closest representative of q1 to q2: \t", q2g)
