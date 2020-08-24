#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['Partial_Solution']

import numpy as np
import copy
import itertools

# A partial solution class.
class Partial_Solution(object):

    # A partial solution is a weighted disjoint union set.
    # The weights are group elements.
    # If i, j are connected, then the weight w[i] * w[j]^{-1} is g_i * g_j^{-1}.

    def __init__(self, num_quats, grp_table, grp_inverse_table):

        self.num_quats = num_quats
        self.fathers = np.arange(0, num_quats, dtype = int)
        self.weights = np.zeros(num_quats, dtype = int)
        self.grp_table = grp_table
        self.grp_inverse_table = grp_inverse_table

        self.cost = 0

    def get_father(self, i):

        if self.fathers[i] == i:
            return i
        else:
            fi = self.fathers[i]
            self.fathers[fi] = self.get_father(fi)
            self.fathers[i] = self.fathers[fi]
            self.weights[i] = self.grp_table[self.weights[i], self.weights[fi]]
            return self.fathers[i]

    def is_connected(self, i, j):

        return self.get_father(i) == self.get_father(j)

    # Return a list of all points connected to i.
    def get_all_connected(self, i):

        return np.where(np.array([self.is_connected(i, j) for j in range(self.num_quats)]))[0]

    # Return w[i]*w[j]^{-1}.
    def distance(self, i, j):

        # i, j must be connected.
        assert(self.get_father(i) == self.get_father(j))

        return self.grp_table[self.weights[i], self.grp_inverse_table[self.weights[j]]]

    # Add a relation g[i] * g[j]^{-1} = g.
    def add_relation(self, i, j, g, f):

        father_i = self.get_father(i)
        father_j = self.get_father(j)

        # i, j must be unconnected
        assert(father_i != father_j)

        cost_weights_i = self.get_all_connected(i)
        cost_weights_j = self.get_all_connected(j)

        self.fathers[father_i] = father_j
        self.weights[father_i] = self.grp_table[self.grp_table[self.grp_inverse_table[self.weights[i]], g], self.weights[j]]

        for weight_i, weight_j in itertools.product(cost_weights_i, cost_weights_j):
            self.cost += f[weight_i, weight_j, self.distance(weight_i, weight_j)]
            self.cost += f[weight_j, weight_i, self.distance(weight_j, weight_i)]

    def copy(self):

        new_ps = copy.deepcopy(self)

        return new_ps
