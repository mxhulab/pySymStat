__all__ = [
    'mean_S2',
    'variance_S2'
]

import numpy as np
from math import comb
from itertools import combinations
from numpy.typing import NDArray
from typing import Literal, Optional

from .distance import distance_S2

def mean_S2(
    vecs : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> NDArray[np.float64]:

    '''
    The `mean_S2` function calculates the mean of a set of projection directions.

    - `vecs`: Unit vector representations of a set of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
    '''

    assert vecs.ndim == 2 and vecs.shape[1] == 3
    if type != 'arithmetic':
        raise NotImplementedError('Only arithmetic distance supported in mean_S2.')

    mean = np.mean(vecs, axis = 0)
    return mean / np.linalg.norm(mean)

def variance_S2(
    vecs : NDArray[np.float64],
    type : Literal['arithmetic'] = 'arithmetic',
    mean : Optional[NDArray[np.float64]] = None
)-> float:

    '''
    The `variance_S2` function calculates the variances of a set of projection directions.

    - `vecs`: Unit vector representations of projection directions, provided as a numpy array with the shape `(n, 3)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input projection directions. If this is `None`, the variance is calculated in an unsupervised manner.
    '''

    assert vecs.ndim == 2 and vecs.shape[1] == 3

    n = len(vecs)
    if mean is None:
        return sum(distance_S2(vecs[i], vecs[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / comb(n, 2)
    else:
        return sum(distance_S2(vecs[i], mean, type = type) ** 2 for i in range(n)) / n
