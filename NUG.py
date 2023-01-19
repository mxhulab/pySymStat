#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['solve_NUG']

import numpy as np
import picos as pic

from .Leaderboard import *
from .Partial_Solution import *
from .SDP import solve_SDP, irreducible_rep_analyze

# apply Peter-Weyl theorem, convert the SDP relaxation solutions into the probability.
def convert_SDP_solutions_to_convex_combination(f, X, grp_irreducible_rep, num_elems_grp):

    num_irreducible_reps = grp_irreducible_rep.shape[0]

    # get dimensions of each irreducible representations
    # get whether or irreducible representations are real matrices or not
    d, is_all_real = irreducible_rep_analyze(grp_irreducible_rep)

    # Get probabilities Y.
    Y = np.zeros_like(f, dtype = X[0].dtype)

    for (i, j, g), _ in np.ndenumerate(Y):
        for k in range(num_irreducible_reps):
            Y[i, j, g] += d[k] * np.trace(grp_irreducible_rep[k, g].conj().transpose() @
                                          X[k][i * d[k] : (i + 1) * d[k], j * d[k] : (j + 1) * d[k]])
        Y[i, j, g] /= num_elems_grp

    if not is_all_real:
        Y = Y.real

    return Y

def recovery_solutions_from_convex_combination(leaderboard_capacity, leaderboard_ignore_threshold, f, Y, grp_table, grp_inverse_table):

    num_quats = Y.shape[0]
    num_elems_grp = Y.shape[2]

    assert(Y.shape == (num_quats, num_quats, num_elems_grp))
    assert(f.shape == Y.shape)

    # preparation of leading board algorithm
    # sort all relations according to probabilities
    relations = np.empty(shape = (num_quats * (num_quats - 1) // 2, num_elems_grp),
                         dtype = [('i', int), ('j', int), ('g', int), ('probability', np.float64)])
    t = 0
    for i in range(num_quats):
        for j in range(i + 1, num_quats):
            for g in range(num_elems_grp):
                relations[t, g] = (i, j, g, Y[i, j, g])
            relations[t] = np.sort(relations[t], order = 'probability')[::-1]
            t += 1
    relations = relations[np.argsort(relations[:, 0], order = 'probability')[::-1]]

    # apply leading board algorithm

    leaderboard = Leaderboard(capacity = leaderboard_capacity, \
                              cost_func = lambda x : x.cost, \
                              dtype = Partial_Solution)

    leaderboard.push(Partial_Solution(num_quats = num_quats, \
                                      grp_table = grp_table, \
                                      grp_inverse_table = grp_inverse_table))

    counter = 0

    for t in range(relations.shape[0]):

        (i, j, _, _) = relations[t, 0]

        if leaderboard.data[0].is_connected(i, j):
            continue

        new_leaderboard = Leaderboard(capacity = leaderboard_capacity, \
                                      cost_func = lambda x : x.cost, \
                                      dtype = Partial_Solution)

        total_probability = 0.

        for r in range(num_elems_grp):

            for s in range(leaderboard.size):
                (i, j, g, _) = relations[t, r]
                new_partial_solution = leaderboard.data[s].copy()
                new_partial_solution.add_relation(i, j, g, f)
                new_leaderboard.push(new_partial_solution)

            total_probability += relations[t, r]['probability']
            if total_probability > leaderboard_ignore_threshold:
                break

        leaderboard = new_leaderboard

        counter += 1

        if counter == num_quats - 1:
            break

    # get solutions

    ## keep only the best one
    while leaderboard.size > 1:
        leaderboard.pop()

    ## get value out
    optimal_solution = np.array([leaderboard.data[0].distance(i, 0) for i in range(num_quats)])
    optimal_cost = leaderboard.data[0].cost

    return optimal_cost, optimal_solution
  
def solve_NUG(f, grp_table, grp_irreducible_rep, leaderboard_capacity = 1, leaderboard_ignore_threshold = 0.99):
    '''
    Solve the following non-unique games over a finite group G:

        min_{g_1,...,g_n in G}sum_{i,j}f_{i,j}(g_ig_j^{-1})

    Here G is presented in an abstract way, i.e., no concrete
    group elements is given. All elements in G is represented
    by an integer between 0 and N-1, where N is the size of
    group G. The group multiplication is given by a multiplication
    table. We require that the identity element is element 0.

    Then the information of all irreducible representations
    (irreps) of G is given by grp_irreducible_rep. The details of its format
    is described below. We note that all irreps should be unitary.

    Parameters
    ==========
    f : a numpy.ndarray, the functions
        f.shape = (n, n, N)
        f.dtype = float type
        f[i, j] represents the function f_{i,j}
        f[i, j, g] := f_{i,j}(G[g]), where
            G[g] means gth element of G
    grp_table : a numpy.ndarray, the multiplication table of G
        grp_table.shape = (N, N)
        grp_table.dtype = int
        G[grp_table[g1, g2]] = G[g1]*G[g2]
    grp_irreducible_rep : a numpy.ndarray, irreps of G
        grp_irreducible_rep.shape = (K, N)
        grp_irreducible_rep.dtype = numpy.ndarray
        grp_irreducible_rep[k, g] : a numpy.ndarray, the kth irrep of G[g]
            grp_irreducible_rep[k, g].shape = (d[k], d[k]), where
                d[k] is the dimension of kth irrep of G
            grp_irreducible_rep[k, g].dtype = np.float64 or np.complex128,
                determined by whether this irrep is real or complex
    leaderboard_capacity : int, 1 by default, board size of leading board
    threshold: float, 0.99 by default, to control how many relations tried.

    Returns
    =======
    (optimal_cost, optimal_solution), where
        optimal_cost : float type, the optimal value of NUG.
        optimal_solution : a numpy.ndarray, the optimal solution of NUG.
            optimal_solution.shape = (n,)
            optimal_solution.dtype = int
            optimal_solution[i] is the index of g_i
    We note that two solutions of NUG are intrinsically the same
    if they are differed by right multiplication of a common group
    element. We retuan a solution such that optimal_solution[0] = 0 is
    the identity element.

    '''
    # n -> num_quats
    # N -> num_elems_grp
    # K -> num_irreducible_reps
    num_quats = f.shape[0]
    num_elems_grp = grp_table.shape[0]
    num_irreducible_reps = grp_irreducible_rep.shape[0]

    assert(f.shape == (num_quats, num_quats, num_elems_grp))
    assert(grp_table.shape == (num_elems_grp, num_elems_grp))

    # get dimensions of each irreducible representations
    # get whether or irreducible representations are real matrices or not
    d, is_all_real = irreducible_rep_analyze(grp_irreducible_rep)

    # solve the SDP problem relaxed from NUG problem
    X = solve_SDP(f, grp_irreducible_rep)

    # convert the solution of the SDP relaxation into probability
    Y = convert_SDP_solutions_to_convex_combination(f, X, grp_irreducible_rep, num_elems_grp)

    # compute table of inverse from grp_table
    grp_inverse_table = np.empty(num_elems_grp, dtype = int)
    for (i_g1, i_g2), g in np.ndenumerate(grp_table):
        if g == 0:
            grp_inverse_table[i_g1] = i_g2

    # recovery true solutions from probability
    return recovery_solutions_from_convex_combination(leaderboard_capacity, leaderboard_ignore_threshold, f, Y, grp_table, grp_inverse_table)
