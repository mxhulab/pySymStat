import numpy as np
from functools import reduce

def _quat_same(q1, q2):
    assert q1.shape[0] == q2.shape[0] == 4
    return np.all(np.square(np.tensordot(q1, q2, axes = (0, 0))) > 1 - 1e-6)

def _quat_mult(q1, q2):
    assert q1.shape[0] == q2.shape[0] == 4
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    ], dtype = np.float64)

def generate_by_two(qa, qb, elems, elems_ab, q, ab):
    '''Generate molecular group (in unit quaternion form) by two generators recursively.
    '''
    if not any(_quat_same(q, q1) for q1 in elems):
        elems.append(q)
        elems_ab.append(ab)
        generate_by_two(qa, qb, elems, elems_ab, _quat_mult(q, qa), ab + 'a')
        generate_by_two(qa, qb, elems, elems_ab, _quat_mult(q, qb), ab + 'b')
        return elems, elems_ab

def orthonormalize(rho):
    n = rho.shape[0]
    d = rho[0].shape[0]
    A = np.zeros((d, d), dtype = np.float64)
    for i, j in np.ndindex(d, d):
        A[i, j] = sum(np.vdot(rho[g][:, i], rho[g][:, j]) for g in range(n)) / n
    V = np.linalg.cholesky(A).T
    Vinv = np.linalg.inv(V)
    for i in range(n):
        rho[i] = V @ rho[i] @ Vinv

def generate_irreps(rhos_ab, elems_ab):
    N = len(elems_ab)
    rhos = []
    for a, b in rhos_ab:
        dk = a.shape[0]
        rho_k = np.empty((N, dk, dk), dtype = np.float64)
        for g in range(N):
            m = map(lambda ch : a if ch == 'a' else b, elems_ab[g])
            rho_k[g] = reduce(lambda x, y : x @ y, m, np.eye(dk, dtype = np.float64))
        orthonormalize(rho_k)
        rhos.append(rho_k)
    return rhos
