__all__ = [
    'mean_SO3',
    'variance_SO3'
]

import numpy as np
from math import comb
from itertools import combinations
from numpy.typing import NDArray
from typing import Literal, Optional

from .distance import distance_SO3

def mean_SO3(
    quats : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> NDArray[np.float64]:

    '''
    The `mean_SO3` function calculates the mean of a set of spatial rotations.

    - `quats`: Unit quaternion representations of a set of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
    '''

    assert quats.ndim == 2 and quats.shape[1] == 4
    if type != 'arithmetic':
        raise NotImplementedError('Only arithmetic distance supported in mean_S2.')

    t = np.mean(quats[:, :, np.newaxis] * quats[:, np.newaxis, :], axis = 0)
    _, eigvects = np.linalg.eigh(t)
    return eigvects[:, -1].copy()

def variance_SO3(
    quats : NDArray[np.float64],
    type : Literal['arithmetic'] = 'arithmetic',
    mean : Optional[NDArray[np.float64]] = None
) -> float:

    '''
    The `variance_SO3` function calculates the variances of a set of spatial rotations.

    - `quats`: Unit quaternion representations of spatial rotations, provided as a numpy array with the shape `(n, 4)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input spatial rotations. If this is `None`, the variance is calculated in an unsupervised manner.
    '''

    assert quats.ndim == 2 and quats.shape[1] == 4

    n = len(quats)
    if mean is None:
        return sum(distance_SO3(quats[i], quats[j], type = type) ** 2 for i, j in combinations(range(n), 2)) / comb(n, 2)
    else:
        return sum(distance_SO3(quats[i], mean, type = type) ** 2 for i in range(n)) / n
