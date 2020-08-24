#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['solve_SDP', \
           'irreducible_rep_analyze']

import itertools
import numpy as np
import picos as pic

def irreducible_rep_analyze(grp_irreducible_rep):

    # compute dimensions of each irreducible representations
    d = np.array([x.shape[0] for x in grp_irreducible_rep[:, 0]])

    # determine whether all irreducible representations are real matrices or not
    is_all_real = np.all(np.array([x.dtype == np.float64 for x in grp_irreducible_rep[:, 0]]))

    return d, is_all_real

def solve_SDP(f, grp_irreducible_rep):

    num_quats = f.shape[0]
    num_elems_grp = f.shape[2]
    num_irreducible_reps = grp_irreducible_rep.shape[0]

    d, is_all_real = irreducible_rep_analyze(grp_irreducible_rep)

    # convert grp_irreducible_rep into picos parameters dim and rho.

    ## dim
    dim = [pic.new_param('d[{k}]'.format(k = k), x) for k, x in enumerate(d)]

    ## rho
    rho = [[pic.new_param('rho[{k}, {g}]'.format(k = k, g = g), x) for g, x in enumerate(y)] for k, y in enumerate(grp_irreducible_rep)]

    # convert grp_irreducible_rep and f inot picos parameters C

    C = num_irreducible_reps * [None]

    for k in range(num_irreducible_reps):

        Ck = np.zeros((num_quats * d[k], num_quats * d[k]), dtype = np.float64 if is_all_real else np.complex128)

        for (i, j, g), _ in np.ndenumerate(f):
            Ck[i * d[k] : (i + 1) * d[k], j * d[k] : (j + 1) * d[k]] += f[j, i, g] * grp_irreducible_rep[k, g].conj().transpose()

        Ck *= (d[k] / num_elems_grp)

        C[k] = pic.new_param('C[{k}]'.format(k = k), Ck)

    # declare PICOS problem
    problem = pic.Problem()

    # construct variables X.
    # note that X[0] is fixed by constraints.
    # so we only construct X[1] .. X[K-1].

    X = num_irreducible_reps * [None]

    for k in range(1, num_irreducible_reps):

        X[k] = problem.add_variable('X[{k}]'.format(k = k), \
                                    (num_quats * d[k], num_quats * d[k]), \
                                    vtype = 'symmetric' if is_all_real else 'hermitian')

    # add constraints

    # add semi-positive defineness constraint
    problem.add_list_of_constraints([X[k] >> 0 for k in range(1, num_irreducible_reps)])

    # add identity matrix constraint
    problem.add_list_of_constraints([pic.diag_vect(X[k]) == 1 for k in range(1, num_irreducible_reps)])

    # add NUG relaxed constraints
    for (i, j, g), _ in np.ndenumerate(f):

        if is_all_real:

            problem.add_constraint(pic.sum([dim[k] * \
                                            pic.trace(rho[k][g] * X[k][i * dim[k] : (i + 1) * dim[k], \
                                                                       j * dim[k] : (j + 1) * dim[k]]) \
                                            for k in range(1, num_irreducible_reps)]) >= -1)

        else:

            problem.add_constraint(pic.sum([dim[k] * \
                                            pic.trace(rho[k][g] * X[k][i * dim[k] : (i + 1) * dim[k], \
                                                                       j * dim[k] : (j + 1) * dim[k]]) \
                                            for k in range(1, num_irreducible_reps)]).real >= -1)

            problem.add_constraint(pic.sum([dim[k] * \
                                            pic.trace(rho[k][g] * X[k][i * dim[k] : (i + 1) * dim[k], \
                                                                       j * dim[k] : (j + 1) * dim[k]]) \
                                            for k in range(1, num_irreducible_reps)]).imag >= 0)
            problem.add_constraint(pic.sum([dim[k] * \
                                            pic.trace(rho[k][g] * X[k][i * dim[k] : (i + 1) * dim[k], \
                                                                       j * dim[k] : (j + 1) * dim[k]]) \
                                            for k in range(1, num_irreducible_reps)]).imag <= 0)

    # set objective

    if is_all_real:

        obj = pic.sum([pic.trace(C[k] * X[k]) for k in range(1, num_irreducible_reps)])
        problem.set_objective('min', obj)

    else:
        obj = pic.sum([pic.trace(C[k] * X[k]) for k in range(1, num_irreducible_reps)]).real
        problem.set_objective('min', obj)

        problem.add_constraint(pic.sum([pic.trace(C[k] * X[k]) for k in range(1, num_irreducible_reps)]).imag >= 0)
        problem.add_constraint(pic.sum([pic.trace(C[k] * X[k]) for k in range(1, num_irreducible_reps)]).imag <= 0)

    # solve

    problem.set_option('verbose', 0)

    problem.solve(solver = 'cvxopt')

    # get X

    X[0] = np.ones((num_quats, num_quats), dtype = np.float64 if is_all_real else np.complex128)
    for k in range(1, num_irreducible_reps):
        X[k] = np.array(X[k].value, dtype = np.float64 if is_all_real else np.complex128)

    return X
