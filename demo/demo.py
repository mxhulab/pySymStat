import sys

sys.path.append('../../')

import numpy as np

from pySymStat import *

if __name__ == '__main__':

    sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep = get_sym_grp('D7')

    print(sym_grp_elems)
    print(sym_grp_table)
    print(sym_grp_irreducible_rep)

    spatial_rotations = np.random.randn(100, 4)
    spatial_rotations = np.apply_along_axis(lambda x : x / np.linalg.norm(x), -1, spatial_rotations)

    print(spatial_rotations)

    mean, variance, representatives, sym_grp_rep = spatial_rotation_mean_variance_SO3_G(spatial_rotations, sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep)

    print(mean)
    print(variance)
    print(representatives)
    print(sym_grp_rep)

    projection_directions = np.random.randn(100, 3)
    projection_directions = np.apply_along_axis(lambda x : x / np.linalg.norm(x), -1, projection_directions)

    print(projection_directions)

    mean, variance, representatives, sym_grp_rep = projection_direction_mean_variance_S2_G(projection_directions, sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep)

    print(mean)
    print(variance)
    print(representatives)
    print(sym_grp_rep)
