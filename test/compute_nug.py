import argparse
import numpy as np
from math import *

import sys
sys.path.append('../../')
from pySymStat import get_sym_grp
from pySymStat.quaternion import quat_mult, quat_rotate, quat_conj
from pySymStat.meanvar import meanvar_SO3_G, meanvar_S2_G

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test program of pySymStat.')
    parser.add_argument('space', choices = ['SO3', 'S2'],               help = 'the space.')
    parser.add_argument('type',  choices = ['arithmetic', 'geometric'], help = 'the distance type.')
    parser.add_argument('group',                                        help = 'the symmetry group.')
    parser.add_argument('--capacity',  type = int,   default = 20,      help = 'the hyperparameter m.')
    parser.add_argument('--threshold', type = float, default = 0.99,    help = 'the hyperparameter c.')
    parser.add_argument('--origin',    action = 'store_true',           help = 'use original rounding algorithm, only works for cyclic groups!')
    args = parser.parse_args()

    d = 3 if args.space == 'S2' else 4
    dataset = np.load(f'data/{args.space}.npy')
    sym_grp_info  = get_sym_grp(args.group)
    sym_grp_elems = sym_grp_info[0]
    n_test = 1000
    n_size = ceil(log(1000, len(sym_grp_elems))) + 1
    grp_action    = quat_mult     if d == 4 else lambda v, q: quat_rotate(quat_conj(q), v)
    meanvar_func  = meanvar_SO3_G if d == 4 else meanvar_S2_G

    # Compute NUG solutions.
    nug_sols = np.empty((n_test, n_size), dtype = np.int32)
    nug_costs = np.empty(n_test, dtype = np.float64)
    n_elems = len(sym_grp_elems)

    for i_test in range(n_test):
        i_dataset = dataset[i_test * n_size : (i_test + 1) * n_size]
        _, nug_cost, _, nug_sol = meanvar_func(
            i_dataset,
            sym_grp_info,
            type = args.type,
            leaderboard_capacity = args.capacity,
            ignore_threshold = args.threshold,
            original_rounding = args.origin
        )
        nug_sols[i_test] = nug_sol
        nug_costs[i_test] = nug_cost
        print(f'Test case {i_test + 1}, nug_sol:', nug_sols[i_test])

    rounding = 'ori' if args.origin else f'm{args.capacity}_c{args.threshold:.2f}'
    np.savez(f'data/nug_{rounding}_{args.space}_{args.type[:3]}_{args.group}.npz', sols = nug_sols, costs = nug_costs)
