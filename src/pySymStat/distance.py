__all__ = [
    'distance_SO3',
    'distance_S2',
    'distance_SO3_G',
    'distance_S2_G'
]

import numpy as np
from math import acos, sqrt
from numpy.typing import NDArray
from typing import Literal, Tuple

from .quaternion import *

def stable_acos(x : float) -> float:
    return acos(max(-1., min(1., x)))

def stable_sqrt(x : float) -> float:
    return sqrt(max(0., x))

def distance_SO3(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> float:

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

def distance_S2(
    v1 : NDArray[np.float64],
    v2 : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> float:

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

from .symmetry_group import get_sym_grp

def distance_SO3_G(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64],
    sym_grp : str,
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> Tuple[float, NDArray[np.float64]]:
    '''
    The `distance_SO3_G` function calculates either the arithmetic or geometric distance between two spatial rotations with molecular symmetry.
    - `q1`, `q2`: These are the unit quaternion representations of spatial rotations, each a Numpy vector of type `np.float64` with a length of 4.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.

    Output:
    - `output[0]`: The distance between two spatial rotations with molecular symmetry.
    - `output[1]`: The closest representative of `q1` to `q2`
    '''

    sym_grp_elems, _ , _ = get_sym_grp(sym_grp) if isinstance(sym_grp, str) else sym_grp 

    g_q1 = np.apply_along_axis(lambda elem: quat_mult(elem, q1), 1, sym_grp_elems)
    ds = np.apply_along_axis(lambda gq: distance_SO3(gq, q2), 1, g_q1)

    min_index = np.argmin(ds)

    return ds[min_index], g_q1[min_index]

def distance_S2_G(
    v1 : NDArray[np.float64],
    v2 : NDArray[np.float64],
    sym_grp : str,
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> Tuple[float, NDArray[np.float64]]:
    '''
    The `distance_S2_G` function calculates either the arithmetic or geometric distance between two projection directions with molecular symmetry.
    - `v1`, `v2`: These are the unit vectors representing projection directions, each a Numpy vector of type `np.float64` with a length of 3.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.

    Output:
    - `output[0]`: The distance between two projection direction with molecular symmetry.
    - `output[1]`: The closest representative of `v1` to `v2`
    '''

    sym_grp_elems, _ , _ = get_sym_grp(sym_grp) if isinstance(sym_grp, str) else sym_grp 

    g_v1 = np.apply_along_axis(lambda elem: quat_rotate(elem, v1), 1, sym_grp_elems)
    ds = np.apply_along_axis(lambda gv: distance_S2(gv, v2), 1, g_v1)

    min_index = np.argmin(ds)

    return ds[min_index], g_v1[min_index]
