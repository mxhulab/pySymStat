__all__ = [
    'Partial_Solution'
]

import numpy as np
from itertools import product

class Partial_Solution(object):
    '''
    A partial solution is a weighted disjoint union set.
    The weights are group elements.
    If i, j are connected, then the weight w[i]*w[j]^{-1} is g_i*g_j^{-1}.
    '''

    def __init__(self, n, grp_table, grp_inv_table, f):
        self.n = n
        self.grp_table = grp_table
        self.grp_inv_table = grp_inv_table
        self.f = f

        self.fathers = np.arange(0, n, dtype = np.int32)
        self.weights = np.zeros(n, dtype = np.int32)
        self.cost = 0.

    def get_father(self, i):
        if self.fathers[i] == i:
            return i
        else:
            fi = self.fathers[i]
            self.fathers[i] = self.get_father(fi)
            self.weights[i] = self.grp_table[self.weights[i], self.weights[fi]]
            return self.fathers[i]

    def is_connected(self, i, j):
        return self.get_father(i) == self.get_father(j)

    # Return w[i]*w[j]^{-1}.
    def get_weight(self, i, j):
        fi = self.get_father(i)
        fj = self.get_father(j)
        assert fi == fj
        return self.grp_table[self.weights[i], self.grp_inv_table[self.weights[j]]]

    # Add a relation g[i]*g[j]^{-1} = g.
    def add_relation(self, i, j, g):
        fi = self.get_father(i)
        fj = self.get_father(j)
        assert fi != fj

        conn_i = [k for k in range(self.n) if self.is_connected(i, k)]
        conn_j = [k for k in range(self.n) if self.is_connected(j, k)]
        self.fathers[fi] = fj
        self.weights[fi] = self.grp_table[self.grp_table[self.grp_inv_table[self.weights[i]], g], self.weights[j]]
        for ii, jj in product(conn_i, conn_j):
            self.cost += self.f[ii, jj, self.get_weight(ii, jj)]
            self.cost += self.f[jj, ii, self.get_weight(jj, ii)]

    def copy(self):
        new_ps = Partial_Solution(self.n, self.grp_table, self.grp_inv_table, self.f)
        new_ps.fathers[:] = self.fathers
        new_ps.weights[:] = self.weights
        new_ps.cost = self.cost
        return new_ps
