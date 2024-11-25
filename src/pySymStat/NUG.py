__all__ = [
    'solve_NUG'
]

import numpy as np
import picos as pic
from itertools import combinations
from numpy.typing import NDArray
from .Leaderboard import Leaderboard
from .PartialSolution import PartialSolution
from .symmetry import Symmetry

def solve_NUG(
    f : NDArray[np.float64],
    sym : Symmetry,
    leaderboard_capacity : int = 20,
    ignore_threshold : float = 0.99,
    verbosity : int = 0
) -> NDArray[np.int32]:
    '''Solve non-unique games (NUG) problem over a finite group G.

    min_{g_1,...,g_n in G}sum_{i,j}f_{i,j}(g_ig_j^{-1})

    Here G is presented in an abstract way, i.e., no concrete
    group elements is given. All elements in G is represented
    by an integer between 0 and M-1, where M is the size of
    group G. The group multiplication is given by a multiplication
    table. We require that the identity element is element 0.

    Then the information of all irreducible representations (irreps)
    of G is given by irreps. The details of its format is described
    below. We note that all irreps should be real orthogonal.

    Parameters
    ----------
    f : array of shape (M, N, N)
        f[g, i, j] is the cost f_{i,j}(g), where M=|G|.
    sym : Symmetry
        Molecular symmetry group.
    leaderboard_capacity : int
        Capacity of the leading board, default is 20.
    ignore_threshold : float
        The threshold to control the how many relations tried,
        default is 0.99.

    Returns
    -------
    solutions : array of shape (N, )
        The optimal solutions {g_i}. Notice that two solutions of NUG
        are intrinsically the same if they are differed by right
        multiplication of a common group element. We retuan a solution
        such that solutions[0] = 0, which is the identity element.
    '''
    N, M = f.shape[1], f.shape[0]
    assert f.shape == (M, N, N)

    irreps = sym.irreps
    splits = sym.splits
    K = len(irreps)
    d = [rho.shape[1] for rho in irreps]
    block = lambda i, j, k : (slice(i * d[k], (i + 1) * d[k]), slice(j * d[k], (j + 1) * d[k]))

    # Compute Fourier transform and obtain picos contant C.
    C = []
    for k in range(K):
        Ck = np.tensordot(f, irreps[k], axes = (0, 0))
        Ck = np.transpose(Ck, axes = (1, 3, 0, 2))
        Ck = np.reshape(Ck, (N * d[k], N * d[k])) * (d[k] / splits[k] * M)
        Ck = (Ck + Ck.T) / 2
        C.append(pic.Constant(f'C[{k}]', Ck))

    # Add variables X.
    # Note that X[0] is fixed by constraints.
    # So we only add X[1] .. X[K-1].
    X = [None] + [pic.SymmetricVariable(f'X[{k}]', (N * d[k], N * d[k])) for k in range(1, K)]

    # Create problems.
    problem = pic.Problem()
    problem.add_list_of_constraints([X[k] >> 0 for k in range(1, K)])
    problem.add_list_of_constraints([pic.maindiag(X[k]) == 1 for k in range(1, K)])
    for g, i, j in np.ndindex(M, N, N):
        problem.add_constraint(pic.sum([d[k] / splits[k] * pic.trace(irreps[k][g] * X[k][block(i, j, k)]) for k in range(1, K)]) >= -1)

    # Set objective and solve.
    objective = pic.sum([pic.trace(C[k] * X[k]) for k in range(1, K)])
    problem.set_objective('min', objective)
    problem.set_option('verbosity', verbosity)
    problem.set_option('max_iterations', 100)   # For some singular cases.
    problem.solve(solver = 'cvxopt')

    # Get probabilities Y.
    X = [np.ones((N, N), dtype = np.float64) if k == 0 else np.array(X[k].value, dtype = np.float64) for k in range(K)]
    Y = 0
    for k in range(K):
        Xk = np.reshape(X[k], (N, d[k], N, d[k]))
        Y += np.tensordot(irreps[k], Xk, axes = ([1, 2], [1, 3])) * (d[k] / splits[k])
    Y /= M

    # Prepare for leader board algorithm.
    # Sort all relations according to probabilities.
    relations = np.empty(shape = (N * (N - 1) // 2, M),
                         dtype = [('i', np.int32), ('j', np.int32), ('g', np.int32), ('probability', np.float64)])
    for t, (i, j) in enumerate(combinations(range(N), 2)):
        for g in range(M): relations[t, g] = (i, j, g, Y[g, i, j])
        relations[t] = np.sort(relations[t], order = 'probability')[::-1]
    relations = relations[np.argsort(relations[:, 0], order = 'probability')[::-1]]

    # apply leader board algorithm
    leaderboard = Leaderboard(leaderboard_capacity, lambda x : x.cost, PartialSolution)
    leaderboard.push(PartialSolution(N, sym.g_table, sym.i_table, f))

    for t in range(len(relations)):
        # Handle relations g_i*g_j^{-1}.

        # If it can be derived from known relations, then ignore it.
        i, j, _, _ = relations[t, 0]
        if leaderboard.data[0].is_connected(i, j): continue

        # Otherwise, try leading relations until ignore_threshold is archieved.
        new_leaderboard = Leaderboard(leaderboard_capacity, lambda x : x.cost, PartialSolution)
        total_probability = 0.
        for r in range(M):
            i, j, g, probability = relations[t, r]
            for s in range(leaderboard.size):
                new_PartialSolution : PartialSolution = leaderboard.data[s].copy()
                new_PartialSolution.add_relation(i, j, g)
                new_leaderboard.push(new_PartialSolution)
            total_probability += probability
            if total_probability > ignore_threshold: break
        leaderboard = new_leaderboard

    # get solutions
    while leaderboard.size > 1: leaderboard.pop()
    return np.array([leaderboard.data[0].get_weight(i, 0) for i in range(N)], dtype = np.int32)

if __name__ == '__main__':
    N, M = 3, 7
    sym = Symmetry(f'C{M}')

    # Solve the following equations:
    # x[i] - x[j] = ? (mod 4)
    x = np.random.randint(M, size = N)
    f = np.random.rand(M, N, N)
    for t, i, j in np.ndindex(M, N, N):
        if (x[i] + M - x[j]) % M == t:
            f[t, i, j] = 0
    sol = solve_NUG(f, sym)
    print((x + M - x[0]) % M)
    print(sol)
