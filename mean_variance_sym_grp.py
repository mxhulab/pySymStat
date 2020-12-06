#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import itertools

from .NUG import solve_NUG
from .quaternion_alg import *
from .symmetry_group import get_sym_grp

__all__ = ['spatial_rotation_mean_variance_SO3_G', \
           'projection_direction_mean_variance_S2_G']

def mean_variance_sym_grp(sr_or_pd, \
                          sym_grp_elems, \
                          sym_grp_table, \
                          sym_grp_irreducible_rep, \
                          num_sr_or_pd_selected_for_NUG, \
                          leaderboard_capacity, \
                          leaderboard_ignore_threshold, \
                          distance_sr_or_pd, \
                          distance_sr_or_pd_G, \
                          mean_sr_or_pd, \
                          grp_operator):

    selected_IDs = np.random.permutation(np.arange(sr_or_pd.shape[0], dtype = np.int))[:num_sr_or_pd_selected_for_NUG]

    sr_or_pd_selected_for_NUG = sr_or_pd[selected_IDs]

    num_quats = sr_or_pd_selected_for_NUG.shape[0]
    num_elems_grp = sym_grp_elems.shape[0]

    # generate f

    f = np.empty((num_quats, num_quats, num_elems_grp), dtype = np.float64)

    for (i, j, g), _ in np.ndenumerate(f):
        f[i, j, g] = distance_sr_or_pd(grp_operator(sr_or_pd_selected_for_NUG[i], sym_grp_elems[g]), sr_or_pd_selected_for_NUG[j])
    
    # print(f)

    # nug

    optimal_cost_selected_for_NUG, optimal_solution_selected_for_NUG = solve_NUG(f, sym_grp_table, sym_grp_irreducible_rep, leaderboard_capacity, leaderboard_ignore_threshold)

    # calculate mean

    sym_grp_elems_solved_selected_for_NUG = np.array([sym_grp_elems[i] for i in optimal_solution_selected_for_NUG])
    representative_solved_selected_for_NUG = np.array([grp_operator(q, g) for q, g in zip(sr_or_pd_selected_for_NUG, sym_grp_elems_solved_selected_for_NUG)])

    mean = mean_sr_or_pd(representative_solved_selected_for_NUG)

    # determine representative_solved and sym_grp_elems_solved_selected_for_NUG

    representative_solved = np.empty(sr_or_pd.shape, dtype = np.float64)
    sym_grp_elems_solved = np.empty((sr_or_pd.shape[0], 4), dtype = np.float64)

    for i in range(sr_or_pd.shape[0]):
        _, representative_solved[i], sym_grp_elems_solved[i] = distance_sr_or_pd_G(sr_or_pd[i], mean, sym_grp_elems)

    # calculate variance

    variance = 0
    for q1, q2 in itertools.product(representative_solved, repeat = 2):
        variance += distance_sr_or_pd(q1, q2)

    return mean, variance, representative_solved, sym_grp_elems_solved

from .distance_SO3_G import distance_SO3, distance_SO3_G
from .spatial_rotation_mean_SO3 import spatial_rotation_mean_SO3

def spatial_rotation_mean_variance_SO3_G(spatial_rotations, \
                                         sym_grp_elems, \
                                         sym_grp_table, \
                                         sym_grp_irreducible_rep, \
                                         num_spatial_rotations_selected_for_NUG = 10, \
                                         leaderboard_capacity = 10, \
                                         leaderboard_ignore_threshold = 0.99):

    return mean_variance_sym_grp(spatial_rotations, \
                                 sym_grp_elems, \
                                 sym_grp_table, \
                                 sym_grp_irreducible_rep, \
                                 num_spatial_rotations_selected_for_NUG, \
                                 leaderboard_capacity, \
                                 leaderboard_ignore_threshold, \
                                 distance_SO3, \
                                 distance_SO3_G, \
                                 spatial_rotation_mean_SO3, \
                                 quat_mult)

from .distance_S2_G import distance_S2, distance_S2_G
from .projection_direction_mean_S2 import projection_direction_mean_S2

def projection_direction_mean_variance_S2_G(projection_directions, \
                                            sym_grp_elems, \
                                            sym_grp_table, \
                                            sym_grp_irreducible_rep, \
                                            num_projection_directions_selected_for_NUG = 10, \
                                            leaderboard_capacity = 10, \
                                            leaderboard_ignore_threshold = 0.99):

    return mean_variance_sym_grp(projection_directions, \
                                 sym_grp_elems, \
                                 sym_grp_table, \
                                 sym_grp_irreducible_rep, \
                                 num_projection_directions_selected_for_NUG, \
                                 leaderboard_capacity, \
                                 leaderboard_ignore_threshold, \
                                 distance_S2, \
                                 distance_S2_G, \
                                 projection_direction_mean_S2, \
                                 lambda v, g : rotate(quat_conj(g), v))
