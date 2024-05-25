import argparse
import numpy as np
from itertools import product
from math import *

from pySymStat import get_sym_grp, mean_SO3, variance_SO3, mean_S2, variance_S2
from pySymStat.quaternion import quat_mult, quat_rotate, quat_conj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test program of pySymStat.')
    parser.add_argument('space', choices = ['SO3', 'S2'], help = 'the space.')
    parser.add_argument('group',                          help = 'the symmetry group.')
    args = parser.parse_args()

    d = 3 if args.space == 'S2' else 4
    dataset = np.load(f'data/{args.space}.npy')
    sym_grp_info  = get_sym_grp(args.group)
    sym_grp_elems = sym_grp_info[0]
    n_test = 1000
    n_size = ceil(log(1000, len(sym_grp_elems))) + 1
    grp_action    = quat_mult     if d == 4 else lambda v, q: quat_rotate(quat_conj(q), v)
    mean_func     = mean_SO3      if d == 4 else mean_S2
    variance_func = variance_SO3  if d == 4 else variance_S2

    # Compute optimal solutions for original problems by brute-force enumeration.
    # Only available for arithmetic distance.
    opt_sols = np.empty((n_test, n_size), dtype = np.int32)
    opt_costs = np.empty(n_test, dtype = np.float64)
    n_elems = len(sym_grp_elems)

    for i_test in range(n_test):
        i_dataset = dataset[i_test * n_size : (i_test + 1) * n_size]

        # Brute force enumeration.
        opt_cost, opt_solution = np.inf, None
        for solution in product(range(n_elems), repeat = n_size - 1):
            new_dataset = np.empty_like(i_dataset)
            new_dataset[0] = i_dataset[0]
            for i in range(1, n_size):
                new_dataset[i] = grp_action(i_dataset[i], sym_grp_elems[solution[i - 1]])
            mean = mean_func(new_dataset, type = 'arithmetic')
            cost = variance_func(new_dataset, type = 'arithmetic', mean = mean)
            if opt_cost > cost: opt_cost, opt_solution = cost, solution

        opt_sols[i_test, 0] = 0
        opt_sols[i_test, 1:] = opt_solution
        opt_costs[i_test] = opt_cost
        print(f'Test case {i_test + 1}, opt_sol:', opt_sols[i_test])

    np.savez(f'data/opt_{args.space}_ari_{args.group}.npz', sols = opt_sols, costs = opt_costs)
