__all__ = [
    'mean_variance_SO3_G',
]

import numpy as np

from .distance import distance_SO3

from .quaternion import quat_mult, quat_conj, quat_rotate

from .meanvar import meanvar_M_G

from .averaging_SO3 import mean_SO3, variance_SO3

from numpy.typing import NDArray

from typing import Literal

def mean_variance_SO3_G(quats : NDArray[np.float64], \
                        sym_grp, \
                        type : Literal['arithmetic'] = 'arithmetic', \
                        **kwargs):
    '''
    The `mean_variance_SO3_G` function calculates the mean and variance of a set of spatial rotations with molecular symmetry.

    - `quats`: Unit quaternion representations of a set of spatail roations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
     - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
     - output[0]: the mean of these spatail roations
     - output[1]: the variacen of these sptail rotations
     - output[2]: the correct representatives of these sptail rotations
     - output[3]: the index of elements in the symmetry group corresponds to the correct represnetative
    '''

    assert(quats.shape[1] == 4)

    grp_action = quat_mult
    distance = lambda q1, q2: distance_SO3(q1, q2, type = type)
    mean_func = lambda quats: mean_SO3(quats, type = type)
    variance_func = lambda quats: variance_SO3(quats, type = type)

    return meanvar_M_G(quats, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)
