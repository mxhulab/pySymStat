import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import ceil, log

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from pySymStat import get_sym_grp, meanvar_SO3_G, meanvar_S2_G
from pySymStat.quaternion import quat_mult, quat_rotate, quat_conj
from pySymStat.meanvar import mean_SO3, variance_SO3, mean_S2, variance_S2

def parse_argument():
    parser = argparse.ArgumentParser(description = 'Test program of pySymStat.')
    parser.add_argument('space',       choices = ['SO3', 'S2'],               help = 'the space.')
    parser.add_argument('group',                                              help = 'the symmetry group.')
    parser.add_argument('type',        choices = ['arithmetic', 'geometric'], help = 'the distance type.')
    parser.add_argument('--dataset',   required = True,                       help = 'the path of dataset.')
    parser.add_argument('--nugsol',    required = True,                       help = 'the path of nug solutions.')
    parser.add_argument('--appsol',    required = True,                       help = 'the path of approximated solutions.')
    parser.add_argument('--optsol',    required = True,                       help = 'the path of optimal solutions.')
    parser.add_argument('--capacity',  type = int,   default = 20,            help = 'the hyperparameter m.')
    parser.add_argument('--threshold', type = float, default = 0.99,          help = 'the hyperparameter c.')
    parser.add_argument('--origin',    action = 'store_true',                 help = 'use original rounding algorithm, only works for cyclic groups!')
    return parser.parse_args()

def compute_optimal(data, batch_num, batch_size, sym_grp_elems, type, grp_action, mean_func, variance_func, centered, prefix):
    optimal_solutions = np.empty((batch_num, batch_size), dtype = np.int32)
    optimal_costs = np.empty(batch_num, dtype = np.float64)
    n_elems = len(sym_grp_elems)

    for batch_i in range(batch_num):
        batch = data[batch_i * batch_size : (batch_i + 1) * batch_size]

        # Brute force enumeration.
        optimal_cost, optimal_solution = np.inf, None
        for solution in product(range(n_elems), repeat = batch_size - 1):
            new_batch = np.empty_like(batch)
            new_batch[0] = batch[0]
            for i in range(1, batch_size):
                new_batch[i] = grp_action(batch[i], sym_grp_elems[solution[i - 1]])
            mean = mean_func(new_batch, type = type) if centered else None
            cost = variance_func(new_batch, type = type, ref = mean)
            if optimal_cost > cost: optimal_cost, optimal_solution = cost, solution

        optimal_solutions[batch_i, 0] = 0
        optimal_solutions[batch_i, 1:] = optimal_solution
        optimal_costs[batch_i] = optimal_cost
        print(f'  Test case {batch_i + 1}, {prefix} solution:', optimal_solutions[batch_i])
        sys.stdout.flush()

    return optimal_solutions, optimal_costs

if __name__ == '__main__':
    args = parse_argument()

    # Set global parameters.
    d = 3 if args.space == 'S2' else 4
    grp_action    = quat_mult     if d == 4 else lambda v, q: quat_rotate(quat_conj(q), v)
    mean_func     = mean_SO3      if d == 4 else mean_S2
    variance_func = variance_SO3  if d == 4 else variance_S2
    meanvar_func  = meanvar_SO3_G if d == 4 else meanvar_S2_G
    sym_grp_info  = get_sym_grp(args.group)
    sym_grp_elems = sym_grp_info[0]
    batch_num     = 1000
    batch_size    = ceil(log(1000, len(sym_grp_elems))) + 1

    # Get dataset.
    if not os.path.exists(args.dataset):
        data = np.random.randn(15000, d)
        data /= np.linalg.norm(data, axis = 1)[:, None]
        np.save(args.dataset, data)
    else:
        data = np.load(args.dataset)
        assert data.shape == (15000, d)

    print(f'Test for space {args.space}, symmetry group {args.group}, distance type {args.type}.')

    # Compute NUG solutions.
    if not os.path.exists(args.nugsol):
        nug_solutions = np.empty((batch_num, batch_size), dtype = np.int32)
        nug_costs = np.empty(batch_num, dtype = np.float64)
        for batch_i in range(batch_num):
            batch = data[batch_i * batch_size : (batch_i + 1) * batch_size]
            _, nug_cost, _, nug_solution = meanvar_func(batch, sym_grp_info, type = args.type, leaderboard_capacity = args.capacity, ignore_threshold = args.threshold, original_rounding = args.origin)
            nug_solutions[batch_i] = nug_solution
            nug_costs[batch_i] = nug_cost
            print(f'  Test case {batch_i + 1}, nug solution:', nug_solutions[batch_i])
            sys.stdout.flush()
        np.savez(args.nugsol, solutions = nug_solutions, costs = nug_costs)
    else:
        npz = np.load(args.nugsol)
        nug_solutions = npz['solutions']
        nug_costs = npz['costs']

    # Compute optimal solutions for approximate problems.
    if not os.path.exists(args.appsol):
        app_solutions, app_costs = compute_optimal(data, batch_num, batch_size, sym_grp_elems, args.type, grp_action, mean_func, variance_func, False, 'app')
        np.savez(args.appsol, solutions = app_solutions, costs = app_costs)
    else:
        npz = np.load(args.appsol)
        app_solutions = npz['solutions']
        app_costs = npz['costs']

    # Compute optimal solutions for original problems.
    # Only computable for arithmetic distance.
    if args.type == 'arithmetic':
        if not os.path.exists(args.optsol):
            opt_solutions, opt_costs = compute_optimal(data, batch_num, batch_size, sym_grp_elems, args.type, grp_action, mean_func, variance_func, True, 'opt')
            np.savez(args.optsol, solutions = opt_solutions, costs = opt_costs)
        else:
            npz = np.load(args.optsol)
            opt_solutions = npz['solutions']
            opt_costs = npz['costs']

    # NUG solutions v.s. approximated solutions.
    success = np.all(nug_solutions == app_solutions, axis = 1).sum()
    print(f'  # NUG solution == approximate solution / # test cases {success} / {batch_num} = {success / batch_num * 100:.1f}%.')
    maxgap = np.max(np.abs(nug_costs - app_costs) / app_costs)
    print(f'  Maximal gap between NUG cost and approximate cost = {maxgap * 100:.4f}%.')

    # approximated solutions v.s. optimal solutions, in arithmetic case.
    if args.type == 'arithmetic':
        success = np.all(app_solutions == opt_solutions, axis = 1).sum()
        print(f'  # Approximate solution == optimal solution / # test cases = {success} / {batch_num} = {success / batch_num * 100:.1f}%.')

        # plot curve / upper / lower bound.
        fig, ax = plt.subplots(figsize = (8, 8))
        if args.space == 'S2':
            x_samples = np.linspace(0, 2, 256, endpoint = True)
            y_samples = 1 - (1 - x_samples / 2) ** 2
            ax.plot(x_samples, y_samples, linewidth = 5, label = r'x = $2(1-\|n\|_2)$' + '\n' + r'y = $1-\|n\|_2^2$', color = 'green', linestyle = 'dashed', zorder = 1)
        else:
            x_samples = np.linspace(0, 6, 1024, endpoint = True)
            y1_samples = np.where(x_samples <= 4,      x_samples - x_samples ** 2 / 8, \
                         np.where(x_samples <= 16 / 3, -8 + 4 * x_samples - 3 / 8 * x_samples ** 2, \
                                                       -24 + 9 * x_samples - 3 / 4 * x_samples ** 2))
            y2_samples = x_samples - 1 / 12 * x_samples ** 2
            ax.plot(x_samples, y1_samples, linewidth = 5, label = 'Lower bound', color = 'red', linestyle = 'dashed', zorder = 1)
            ax.plot(x_samples, y2_samples, linewidth = 5, label = 'Upper bound', color = 'blue', linestyle = 'dashed', zorder = 1)

        ax.scatter(opt_costs, app_costs, c = 'black', s = 100, marker = '+', alpha = 0.8, zorder = 2)
        ax.legend(fontsize = 'x-large')
        fig.savefig(f'output.png', dpi = 300)
        plt.close(fig)
