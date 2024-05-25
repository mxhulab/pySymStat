import numpy as np

from .symmetry_group import get_sym_grp

from .NUG import solve_NUG

def meanvar_M_G(data, sym_grp, grp_action, distance, mean_func, variance_func, **kwargs):
    sym_grp_elems, sym_grp_table, sym_grp_irreps = get_sym_grp(sym_grp) if isinstance(sym_grp, str) else sym_grp
    n = len(data)
    N = len(sym_grp_elems)

    # Generate f.
    f = np.empty((n, n, N), dtype = np.float64)
    for i, j, g in np.ndindex(n, n, N):
        f[i, j, g] = distance(grp_action(data[i], sym_grp_elems[g]), data[j]) ** 2

    # Solve NUG and get results.
    solutions = solve_NUG(f, sym_grp_table, sym_grp_irreps, **kwargs)
    representatives = np.empty_like(data, dtype = np.float64)
    for i in range(n):
        representatives[i] = grp_action(data[i], sym_grp_elems[solutions[i]])
    mean = mean_func(representatives)
    var = variance_func(representatives)
    return mean, var, representatives, solutions
