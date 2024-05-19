__all__ = [
    'mean_SO3',
    'variance_SO3'
]

import numpy as np
from itertools import combinations

import math

from .distance import distance_SO3

from numpy.typing import NDArray

from typing import Literal

def mean_SO3(quats : NDArray[np.float64], \
             type : Literal['arithmetic'] = 'arithmetic') -> NDArray[np.float64]:
    '''
    The `mean_SO3` function calculates the mean of a set of spatial rotations.

    - `quats`: Unit quaternion representations of a set of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
    '''
    assert(quats.shape[1] == 4)

    n = len(quats)
    t = np.mean(quats.reshape((n, 4, 1)) * quats.reshape((n, 1, 4)), axis = 0)

    _, eigvects = np.linalg.eigh(t)

    return eigvects[:, -1].copy()

def variance_SO3(quats : NDArray[np.float64], \
                 type : Literal['arithmetic'] = 'arithmetic', \
                 mean : Literal[None, NDArray[np.float64]] = None) -> float:

    assert(quats.shape[1] == 4)

    n = len(quats)

    if mean is None:

        return sum(distance_SO3(quats[i], quats[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / math.comb(n, 2)

    else:

        return sum(distance_SO3(quats[i], mean, type = type) ** 2 for i in range(n)) / n
