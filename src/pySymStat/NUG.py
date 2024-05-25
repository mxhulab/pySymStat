__all__ = [
    'solve_NUG'
]

import numpy as np
import picos as pic
from itertools import combinations
from numpy.typing import NDArray

from .Leaderboard import Leaderboard
from .Partial_Solution import Partial_Solution

def solve_NUG(
    f : NDArray[np.float64],
    grp_table : NDArray[np.int32],
    grp_irreps : NDArray,
    leaderboard_capacity : int = 20,
    ignore_threshold : float = 0.99,
    original_rounding : bool = False,
    verbosity : int = 0
) -> NDArray[np.int32]:

    '''
    Solve the following non-unique games (NUG) over a finite group G:

        min_{g_1,...,g_n in G}sum_{i,j}f_{i,j}(g_ig_j^{-1})

    Here G is presented in an abstract way, i.e., no concrete
    group elements is given. All elements in G is represented
    by an integer between 0 and N-1, where N is the size of
    group G. The group multiplication is given by a multiplication
    table. We require that the identity element is element 0.

    Then the information of all irreducible representations
    (irreps) of G is given by grp_irreps. The details of its format
    is described below. We note that all irreps should be unitary.

    Parameters
    ==========
    f : a numpy.ndarray, the functions.
        f.shape = (n, n, N)
        f.dtype = float type
        f[i, j] represents the function f_{i,j}
        f[i, j, g] := f_{i,j}(G[g]), where
            G[g] means gth element of G.

    grp_table : a numpy.ndarray, the multiplication table of G.
        grp_table.shape = (N, N)
        grp_table.dtype = np.int32
        G[grp_table[g1, g2]] = G[g1]*G[g2]

    grp_irreps : a numpy.ndarray, irreps of G.
        grp_irreps.shape = (K, N)
        grp_irreps.dtype = numpy.ndarray
        grp_irreps[k, g] : a numpy.ndarray, the kth irrep of G[g].
            grp_irreps[k, g].shape = (d[k], d[k]), where
                d[k] is the dimension of kth irrep of G.
            grp_irreps[k, g].dtype = np.float64 or np.complex128,
                determined by whether this irrep is real or complex.

    leaderboard_capacity : int, 1 by default, board size of leading board.

    ignore_threshold: float, 0.99 by default, to control how many relations tried.

    Returns
    =======
    optimal_solution : a numpy.ndarray, the optimal solution of NUG.
        optimal_solution.shape = (n,)
        optimal_solution.dtype = np.int32
        optimal_solution[i] is the index of g_i.

    Notice that two solutions of NUG are intrinsically the same
    if they are differed by right multiplication of a common group
    element. We retuan a solution such that optimal_solution[0] = 0 is
    the identity element.
    '''

    n = f.shape[0]
    N = grp_table.shape[0]
    K = grp_irreps.shape[0]
    assert f.shape == (n, n, N)
    assert grp_table.shape == (N, N)
    assert grp_irreps.shape == (K, N)

    d = [rho[0].shape[0] for rho in grp_irreps]
    all_real = all(rho[0].dtype == np.float64 for rho in grp_irreps)
    dtype = np.float64 if all_real else np.complex128
    block = lambda i, j, k : (slice(i * d[k], (i + 1) * d[k]), slice(j * d[k], (j + 1) * d[k]))

    # Convert irreps into picos constant.
    rho = [[pic.Constant(f'rho[{k}, {g}]', grp_irreps[k, g]) for g in range(N)] for k in range(K)]

    # Compute Fourier transform and obtain picos contant C.
    C = np.empty(K, dtype = object)
    for k in range(K):
        Ck = np.zeros((n * d[k], n * d[k]), dtype = dtype)
        for i, j, g in np.ndindex(n, n, N):
            Ck[block(i, j, k)] += f[j, i, g] * grp_irreps[k, g].conj().transpose()
        Ck *= d[k] / N
        C[k] = pic.Constant(f'C[{k}]', Ck)

    # Add variables X.
    # Note that X[0] is fixed by constraints.
    # So we only add X[1] .. X[K-1].
    X = [None] + [pic.SymmetricVariable(f'X[{k}]', (n * d[k], n * d[k])) if all_real else \
                  pic.HermitianVariable(f'X[{k}]', (n * d[k], n * d[k])) for k in range(1, K)]

    # Create problems.
    problem = pic.Problem()
    problem.add_list_of_constraints([X[k] >> 0 for k in range(1, K)])
    problem.add_list_of_constraints([pic.maindiag(X[k]) == 1 for k in range(1, K)])
    for i, j, g in np.ndindex(n, n, N):
        if all_real:
            problem.add_constraint(pic.sum([d[k] * pic.trace(rho[k][g] * X[k][block(i, j, k)]) for k in range(1, K)]) >= -1)
        else:
            problem.add_constraint(pic.sum([d[k] * pic.trace(rho[k][g] * X[k][block(i, j, k)]) for k in range(1, K)]).real >= -1)
            problem.add_constraint(pic.sum([d[k] * pic.trace(rho[k][g] * X[k][block(i, j, k)]) for k in range(1, K)]).imag <= 0)
            problem.add_constraint(pic.sum([d[k] * pic.trace(rho[k][g] * X[k][block(i, j, k)]) for k in range(1, K)]).imag >= 0)

    # Set objective and solve.
    if all_real:
        objective = pic.sum([pic.trace(C[k] * X[k]) for k in range(1, K)])
    else:
        objective = pic.sum([pic.trace(C[k] * X[k]) for k in range(1, K)]).real
        problem.add_constraint(pic.sum([pic.trace(C[k] * X[k]) for k in range(1, K)]).imag <= 0)
        problem.add_constraint(pic.sum([pic.trace(C[k] * X[k]) for k in range(1, K)]).imag >= 0)
    problem.set_objective('min', objective)
    problem.set_option('verbosity', verbosity)
    problem.set_option('max_iterations', 100)   # For some singular cases.
    problem.solve(solver = 'cvxopt')

    # Get value of X.
    X = [np.ones((n, n), dtype = dtype) if k == 0 else np.array(X[k].value, dtype = dtype) for k in range(K)]

    # Only works for cyclic groups.
    if original_rounding:
        _, eigvecs = np.linalg.eigh(X[1])
        eigvec = eigvecs[:, -1]
        eigvec = eigvec / eigvec[0]
        return np.array(np.round(np.angle(eigvec) / (2 * np.pi) * N) % N, dtype = np.int32)

    # Get probabilities Y.
    Y = np.zeros((n, n, N), dtype = X[0].dtype)
    for i, j, g, k in np.ndindex(n, n, N, K):
        Y[i, j, g] += d[k] * np.trace(grp_irreps[k, g].conj().transpose() @ X[k][block(i, j, k)])
    Y = Y.real / N

    # Prepare for leader board algorithm.
    # Sort all relations according to probabilities.
    relations = np.empty(shape = (n * (n - 1) // 2, N),
                         dtype = [('i', np.int32), ('j', np.int32), ('g', np.int32), ('probability', np.float64)])
    for t, (i, j) in enumerate(combinations(range(n), 2)):
        for g in range(N): relations[t, g] = (i, j, g, Y[i, j, g])
        relations[t] = np.sort(relations[t], order = 'probability')[::-1]
    relations = relations[np.argsort(relations[:, 0], order = 'probability')[::-1]]

    # compute table of inverse from grp_table
    grp_inv_table = np.empty(N, dtype = np.int32)
    for i, j in np.ndindex(N, N):
        if grp_table[i, j] == 0: grp_inv_table[i] = j

    # apply leader board algorithm
    leaderboard = Leaderboard(leaderboard_capacity, lambda x : x.cost, Partial_Solution)
    leaderboard.push(Partial_Solution(n, grp_table, grp_inv_table, f))

    for t in range(len(relations)):
        # Handle relations g_i*g_j^{-1}.

        # If it can be derived from known relations, then ignore it.
        i, j, _, _ = relations[t, 0]
        if leaderboard.data[0].is_connected(i, j): continue

        # Otherwise, try leading relations until ignore_threshold is archieved.
        new_leaderboard = Leaderboard(leaderboard_capacity, lambda x : x.cost, Partial_Solution)
        total_probability = 0.
        for r in range(N):
            i, j, g, probability = relations[t, r]
            for s in range(leaderboard.size):
                new_partial_solution = leaderboard.data[s].copy()
                new_partial_solution.add_relation(i, j, g)
                new_leaderboard.push(new_partial_solution)
            total_probability += probability
            if total_probability > ignore_threshold: break
        leaderboard = new_leaderboard

    # get solutions
    while leaderboard.size > 1: leaderboard.pop()
    return np.array([leaderboard.data[0].get_weight(i, 0) for i in range(n)], dtype = np.int32)

if __name__ == '__main__':
    from .symmetry_group import get_sym_grp
    _, grp_table, grp_irreps = get_sym_grp('C4')
    n = 3
    N = 4

    # Solve the following equations:
    # x0 - x1 = 1 (mod 4)
    # x0 - x2 = 0 (mod 4)
    x = [0, 3, 0]
    f = np.random.rand(3, 3, 4)
    for i, j, t in np.ndindex(3, 3, 4):
        if (x[i] - x[j]) % 4 == t: f[i, j, t] = 0
    print(solve_NUG(f, grp_table, grp_irreps))      # (0, x)
