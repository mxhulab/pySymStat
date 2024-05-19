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
                 type : Literal['arithmetic', 'geometric'] = 'arithmetic') -> float:
    '''
    The `distance_SO3` function calculates either the arithmetic or geometric distance between two spatial rotations.
    - `q1`, `q2`: These are the unit quaternion representations of spatial rotations, each a Numpy vector of type `np.float64` with a length of 4.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
    '''

    if type == 'arithmetic':
        return 2 * stable_sqrt(2 * (1 - np.dot(q1, q2) ** 2))
    elif type == 'geometric':
        return 2 * stable_acos(abs(np.dot(q1, q2)))
    else:
        raise ValueError('Invalid argument.')

def distance_S2(v1 : NDArray[np.float64], \
                v2 : NDArray[np.float64], \
                type : Literal['arithmetic', 'geometric'] = 'arithmetic') -> float:
    '''
    The `distance_S2` function calculates either the arithmetic or geometric distance between two projection directions.
    - `v1`, `v2`: These are the unit vectors representing projection directions, each a Numpy vector of type `np.float64` with a length of 3.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
    '''

    if type == 'arithmetic':
        return np.linalg.norm(v1 - v2)
    elif type == 'geometric':
        return stable_acos(np.dot(v1, v2))
    else:
        raise ValueError('Invalid argument.')
