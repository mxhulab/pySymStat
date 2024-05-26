import matplotlib.pyplot as plt
import numpy as np
import pylab
from itertools import product
from math import *

from pySymStat import get_sym_grp, mean_SO3, variance_SO3
from pySymStat.quaternion import quat_mult

spaces = ['SO3', 'S2']
types  = ['arithmetic', 'geometric']
groups = ['C2', 'C7', 'D2', 'D7', 'T', 'O', 'I']

if __name__ == '__main__':
    print(r'Approximation ability of \tilde{L}^{SO(3)} to L^{SO(3)}.')
    for group in groups:
        opt_npz = np.load(f'data/opt_SO3_ari_{group}.npz')
        app_npz = np.load(f'data/app_SO3_ari_{group}.npz')
        opt_sols = opt_npz['sols']
        app_sols = app_npz['sols']
        opt_costs = opt_npz['costs']
        app_costs = app_npz['costs']

        success = np.sum(np.all(opt_sols == app_sols, axis = 1))
        roe = success / 1000 * 100

        # Compute RCG.
        # Notice that L^{SO(3)}(\tilde{g}_i) != \tilde{L}^{SO(3)}(\tilde{g}_i).
        # We should re-compute app_costs.
        dataset = np.load('data/SO3.npy')
        sym_grp_info  = get_sym_grp(group)
        sym_grp_elems = sym_grp_info[0]
        n_test = 1000
        n_size = ceil(log(1000, len(sym_grp_elems))) + 1
        for i_test in range(n_test):
            i_dataset = dataset[i_test * n_size : (i_test + 1) * n_size].copy()
            for i in range(n_size):
                i_dataset[i] = quat_mult(i_dataset[i], sym_grp_elems[app_sols[i_test, i]])
            mean = mean_SO3(i_dataset, type = 'arithmetic')
            app_costs[i_test] = variance_SO3(i_dataset, type = 'arithmetic', mean = mean)
        rcg = np.abs(opt_costs - app_costs) / opt_costs
        rcg1 = np.sum(rcg < 0.01) / 1000 * 100
        rcg2 = np.sum(rcg < 0.1)  / 1000 * 100

        print(f'  For group {group:2s}, RoE = {roe:.1f}%, Pr[RCG < 0.01] = {rcg1:.1f}%, Pr[RCG < 0.1] = {rcg2:.1f}%.')
    print()


    print(r'Approximation ability of \tilde{L}^{S2} to L^{S2}.')
    print(r'According to THEOREM 2.2, RoE should be 100%.')
    for group in groups:
        opt_npz = np.load(f'data/opt_S2_ari_{group}.npz')
        app_npz = np.load(f'data/app_S2_ari_{group}.npz')
        opt_sols = opt_npz['sols']
        app_sols = app_npz['sols']
        success = np.sum(np.all(opt_sols == app_sols, axis = 1))
        print(f'  For group {group:2s}, RoE = {success / 10:.1f}%.')
    print()


    print(r'Plot scatter point graph of (\tilde{L}^{S^2}, L^{S^2}).')
    pylab.rc('axes', linewidth = 5)
    fig, ax = plt.subplots(figsize = (8, 8))
    x_samples = np.linspace(0, 2, 256)
    y_samples = 1 - (1 - x_samples / 2) ** 2
    ax.plot(x_samples, y_samples, linewidth = 5, label = r'x = $2(1-\|n\|_2)$' + '\n' + r'y = $1-\|n\|_2^2$', color = 'green', linestyle = 'dashed', zorder = 1)
    for group, color in zip(['C2', 'T'], ['black', 'orange']):
        opt_npz = np.load(f'data/opt_S2_ari_{group}.npz')
        app_npz = np.load(f'data/app_S2_ari_{group}.npz')
        opt_sols = opt_npz['sols']
        app_sols = app_npz['sols']
        opt_costs = opt_npz['costs']
        app_costs = app_npz['costs']
        ax.scatter(opt_costs, app_costs, c = color, s = 100, marker = '+', alpha = 0.8, label = '(Cost A, Cost B)', zorder = 2)
    ax.tick_params(axis = 'both', width = 5, length = 10, labelsize = 20)
    fig.savefig(f'data/S2.png', dpi = 300)
    plt.close(fig)

    print(r'Plot scatter point graph of (\tilde{L}^{SO(3)}, L^{SO(3)}).')
    fig, ax = plt.subplots(figsize = (8, 8))
    x_samples = np.linspace(0, 6, 1024)
    y1_samples = np.where(x_samples <= 4,      x_samples - x_samples ** 2 / 8, \
                 np.where(x_samples <= 16 / 3, -8 + 4 * x_samples - 3 / 8 * x_samples ** 2, \
                                               -24 + 9 * x_samples - 3 / 4 * x_samples ** 2))
    y2_samples = x_samples - 1 / 12 * x_samples ** 2
    ax.plot(x_samples, y1_samples, linewidth = 5, label = 'Lower bound', color = 'red', linestyle = 'dashed', zorder = 1)
    ax.plot(x_samples, y2_samples, linewidth = 5, label = 'Upper bound', color = 'blue', linestyle = 'dashed', zorder = 1)
    for group, color in zip(['C2', 'T'], ['black', 'orange']):
        opt_npz = np.load(f'data/opt_SO3_ari_{group}.npz')
        app_npz = np.load(f'data/app_SO3_ari_{group}.npz')
        opt_sols = opt_npz['sols']
        app_sols = app_npz['sols']
        opt_costs = opt_npz['costs']
        app_costs = app_npz['costs']
        ax.scatter(opt_costs, app_costs, c = color, s = 100, marker = '+', alpha = 0.8, label = '(Cost A, Cost B)', zorder = 2)
    ax.tick_params(axis = 'both', width = 5, length = 10, labelsize = 20)
    fig.savefig(f'data/SO3.png', dpi = 300)
    plt.close(fig)
    print()


    print(r'Approximation ability of NUG approach with our rounding for \tilde{L}^{SO(3)}.')
    print(r'Show (RoE, max-RCG).')
    print(r'| Group |   d_{SO(3)}^A   |   d_{SO(3)}^G   |    d_{S2}^A     |    d_{S2}^G     |')
    print(r'|-------------------------------------------------------------------------------|')
    for group in groups:
        print(f'|   {group:2s}  |', end = '')
        for space, type in product(spaces, types):
            app_npz = np.load(f'data/app_{space}_{type[:3]}_{group}.npz')
            nug_npz = np.load(f'data/nug_m20_c0.99_{space}_{type[:3]}_{group}.npz')
            app_sols = app_npz['sols']
            nug_sols = nug_npz['sols']
            app_costs = app_npz['costs']
            nug_costs = nug_npz['costs']

            success = np.sum(np.all(app_sols == nug_sols, axis = 1))
            roe = success / 1000 * 100
            maxrcg = np.max(np.abs(nug_costs - app_costs) / app_costs) * 100
            print(f' ({roe:5.1f}%, {maxrcg:3.2f}%) |', end = '')
        print()
    print(r'|-------------------------------------------------------------------------------|')
    print()


    print(r'Approximation ability of NUG approach with original rounding for \tilde{L}^{SO(3)}.')
    print(r'Show (RoE, max-RCG).')
    print(r'| Group |   d_{SO(3)}^A   |   d_{SO(3)}^G   |    d_{S2}^A     |    d_{S2}^G     |')
    print(r'|-------------------------------------------------------------------------------|')
    for group in ['C2', 'C7']:
        print(f'|   {group:2s}  |', end = '')
        for space, type in product(spaces, types):
            app_npz = np.load(f'data/app_{space}_{type[:3]}_{group}.npz')
            nug_npz = np.load(f'data/nug_ori_{space}_{type[:3]}_{group}.npz')
            app_sols = app_npz['sols']
            nug_sols = nug_npz['sols']
            app_costs = app_npz['costs']
            nug_costs = nug_npz['costs']

            success = np.sum(np.all(app_sols == nug_sols, axis = 1))
            roe = success / 1000 * 100
            maxrcg = np.max(np.abs(nug_costs - app_costs) / app_costs) * 100
            print(f' ({roe:5.1f}%, {maxrcg:3.2f}%) |', end = '')
        print()
    print(r'|-------------------------------------------------------------------------------|')


    print(r'The (RoE, max-RCG) results for varying $m$ under $d_{SO(3)}^A$.')
    print(r'|-------------------------------------------------------------------------------|')
    print(r'| Group |   m=12, c=0.99  |   m=4, c=0.99   |   m=20, c=0.5   |    m=20, c=0    |')
    for group in groups:
        print(f'|   {group:2s}  |', end = '')
        for m, c in [(12, 0.99), (4, 0.99), (20, 0.5), (20, 0)]:
            app_npz = np.load(f'data/app_SO3_ari_{group}.npz')
            nug_npz = np.load(f'data/nug_m{m}_c{c:.2f}_SO3_ari_{group}.npz')
            app_sols = app_npz['sols']
            nug_sols = nug_npz['sols']
            app_costs = app_npz['costs']
            nug_costs = nug_npz['costs']
            success = np.sum(np.all(app_sols == nug_sols, axis = 1))
            roe = success / 1000 * 100
            maxrcg = np.max(np.abs(nug_costs - app_costs) / app_costs) * 100
            print(f' ({roe:5.1f}%, {maxrcg:3.2f}%) |', end = '')
        print()
    print(r'|-------------------------------------------------------------------------------|')
    print()
