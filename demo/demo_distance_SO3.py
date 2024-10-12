import numpy as np
from pySymStat import distance_SO3

if __name__ == '__main__':
    q1 = np.random.randn(4, 2)
    q1 /= np.linalg.norm(q1, axis = 0, keepdims = True)
    q2 = np.random.randn(4, 2)
    q2 /= np.linalg.norm(q2, axis = 0, keepdims = True)
    print("spatial rotation q1: ", q1)
    print("spatial rotation q2: ", q2)
    print("arithmetic distance between q1 and q2: \t", distance_SO3(q1, q2, 'arithmetic'))
    print("geometric distance between q1 and q2: \t" , distance_SO3(q1, q2, 'geometric'))
    print("pairwise arithmetic distance between q1 and q2: \t", distance_SO3(q1, q2, 'arithmetic', True))
    print("pairwise geometric distance between q1 and q2: \t" , distance_SO3(q1, q2, 'geometric', True))
