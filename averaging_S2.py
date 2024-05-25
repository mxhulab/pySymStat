__all__ = [
    'mean_S2',
    'variance_S2'
]

import numpy as np
from itertools import combinations

import math

from .distance import distance_S2

from numpy.typing import NDArray

from typing import Literal

def mean_S2(vecs : NDArray[np.float64], \
            type : Literal['arithmetic'] = 'arithmetic') -> NDArray[np.float64]:
    '''
    The `mean_S2` function calculates the mean of a set of projection directions.

    - `quats`: Unit vector representations of a set of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
    '''

    assert(vecs.shape[1] == 3)

    mean = np.mean(vecs, axis = 0)
    return mean / np.linalg.norm(mean)

def variance_S2(vecs, type = 'arithmetic', ref = None):
    n = len(vecs)
    if ref is None:
        return sum(distance_S2(vecs[i], vecs[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / (n * n)
    else:
        return sum(distance_S2(vecs[i], ref, type = type) ** 2 for i in range(n)) / n
