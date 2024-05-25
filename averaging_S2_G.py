__all__ = [
    'mean_variance_SO3_G',
]

import numpy as np

from .distance import distance_S2

from .quaternion import quat_conj, quat_rotate

from .meanvar import meanvar_M_G

from .averaging_S2 import mean_S2, variance_S2

from numpy.typing import NDArray

from typing import Literal

def mean_variance_S2_G(vecs : NDArray[np.float64], \
                       sym_grp, \
                       type : Literal['arithmetic'] = 'arithmetic', \
                       **kwargs):
    '''
    The `mean_variance_S2_G` function calculates the mean and variance of a set of projection directions with molecular symmetry.

    - `quats`: Unit quaternion representations of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.

    Output:
    - `output[0]`: The mean of these projection directions.
    - `output[1]`: The variance of these projection directions.
    - `output[2]`: The correct representatives of these projection directions.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
    '''

    assert(vecs.shape[1] == 3)

    grp_action = lambda v, q: quat_rotate(quat_conj(q), v)
    distance = lambda v1, v2: distance_S2(v1, v2, type = type)
    mean_func = lambda vecs: mean_S2(vecs, type = type)
    variance_func = lambda vecs: variance_S2(vecs, type = type)

    return meanvar_M_G(vecs, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs)
