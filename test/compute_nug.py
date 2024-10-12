import argparse
import numpy as np
from math import *
from pySymStat import get_grp_info, mean_variance_SO3_G, mean_variance_S2_G, action_SO3, action_S2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test program of pySymStat.')
    parser.add_argument('space', choices = ['SO3', 'S2'],               help = 'the space.')
    parser.add_argument('type',  choices = ['arithmetic', 'geometric'], help = 'the distance type.')
    parser.add_argument('group',                                        help = 'the symmetry group.')
    parser.add_argument('--capacity',  type = int,   default = 20,      help = 'the hyperparameter m.')
    parser.add_argument('--threshold', type = float, default = 0.99,    help = 'the hyperparameter c.')
    args = parser.parse_args()

    d = 4 if args.space == 'SO3' else 3
    dataset = np.load(f'data/{args.space}.npy')
    sym_grp_info  = get_grp_info(args.group)
    sym_grp_elems = sym_grp_info[0]
    n_test = 1000
    n_size = ceil(log(1000, len(sym_grp_elems))) + 1
    grp_action   = action_SO3          if d == 4 else action_S2
    meanvar_func = mean_variance_SO3_G if d == 4 else mean_variance_S2_G

    # Compute NUG solutions.
    nug_sols = np.empty((n_test, n_size), dtype = np.int32)
    nug_costs = np.empty(n_test, dtype = np.float64)
    n_elems = len(sym_grp_elems)

    for i_test in range(n_test):
        i_dataset = dataset[i_test * n_size : (i_test + 1) * n_size]
        _, _, nug_cost, nug_sol, _ = meanvar_func(
            i_dataset,
            sym_grp_info,
            type = args.type,
            leaderboard_capacity = args.capacity,
            ignore_threshold = args.threshold
        )
        nug_sols[i_test] = nug_sol
        nug_costs[i_test] = nug_cost
        print(f'Test case {i_test + 1}, nug_sol:', nug_sols[i_test])

    np.savez(f'data/nug_m{args.capacity}_c{args.threshold:.2f}_{args.space}_{args.type[:3]}_{args.group}.npz', sols = nug_sols, costs = nug_costs)
