__all__ = [
    'mean_variance_SO3_G',
]

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple

from .distance import distance_SO3
from .quaternion import quat_mult
from .meanvar import meanvar_M_G
from .averaging_SO3 import mean_SO3, variance_SO3

def mean_variance_SO3_G(
    quats : NDArray[np.float64],
    sym_grp : str,
    type : Literal['arithmetic', 'geometric'] = 'arithmetic',
    **kwargs
) -> Tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.int32]]:

    '''
    The `mean_variance_SO3_G` function calculates the mean and variance of a set of spatial rotations with molecular symmetry.

    - `quats`: Unit quaternion representations of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.

    Output:
    - `output[0]`: The mean of these spatial rotations.
    - `output[1]`: The variance of these spatial rotations.
    - `output[2]`: The correct representatives of these spatial rotations.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
    '''

    assert quats.ndim == 2 and quats.shape[1] == 4

    grp_action = quat_mult
    distance = lambda q1, q2: distance_SO3(q1, q2, type = type)
    mean_func = lambda quats: mean_SO3(quats, type = type)
    variance_func = lambda quats: variance_SO3(quats, type = type)

    return meanvar_M_G(quats, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)
