__all__ = [
    'distance_SO3',
    'distance_S2'
]

import numpy as np
from math import pi, sin, cos, acos, sqrt, isclose
from .quaternion import *

from numpy.typing import NDArray
from typing import Literal

def stable_acos(x):
    return acos(max(-1., min(1., x)))

def stable_sqrt(x):
    return sqrt(max(0., x))

def distance_SO3(q1 : NDArray[np.float64], \
                 q2 : NDArray[np.float64], \
                 type = Literal['arithmetic', 'geometric']) -> float:
    '''
    q1, q2 : unit quaternion representation of spatial rotations, Numpy vector of np.float64 of length 4.
    type : 'arithmetic' | 'geometric'.
    '''
    if type == 'arithmetic':
        return 2 * stable_sqrt(2 * (1 - np.dot(q1, q2) ** 2))
    elif type == 'geometric':
        return 2 * stable_acos(abs(np.dot(q1, q2)))
    else:
        raise ValueError('Invalid argument.')

def distance_S2(v1, v2, type = 'arithmetic'):
    '''
    v1, v2 : unit 3d vector.
    type : 'arithmetic' | 'geometric'.
    '''
    if type == 'arithmetic':
        return np.linalg.norm(v1 - v2)
    elif type == 'geometric':
        return stable_acos(np.dot(v1, v2))
    else:
        raise ValueError('Invalid argument.')

if __name__ == '__main__':

    q1 = np.random.randn(4)
    q2 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    print('q1 =', q1)
    print('q2 =', q2)

    dA = distance_SO3(q1, q2, type = 'arithmetic')
    dG = distance_SO3(q1, q2, type = 'geometric')
    print('d_A(q1, q2) =', dA)
    print('d_G(q1, q2) =', dG)
    assert 0 <= dA <= 8 ** 0.5
    assert 0 <= dG <= pi
    assert isclose(dA, 8 ** 0.5 * sin(dG / 2))

    print()

    v1 = np.random.randn(3)
    v2 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    print('v1 =', v1)
    print('v2 =', v2)

    dA = distance_S2(v1, v2, type = 'arithmetic')
    dG = distance_S2(v1, v2, type = 'geometric')
    print('d_A(v1, v2) =', dA)
    print('d_G(v1, v2) =', dG)
    assert 0 <= dA <= 2
    assert 0 <= dG <= pi
    assert isclose(dA, sqrt(2 - 2 * cos(dG)))
