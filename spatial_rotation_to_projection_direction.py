__all__ = ['spatial_rotation_to_projection_direction']

import numpy as np

from .quaternion_alg import rotate

def spatial_rotation_to_projection_direction(sr):

    vn = np.array([0, 0, 1], dtype = np.float)

    return rotate(sr, vn)
