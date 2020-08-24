#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['Leaderboard']

class Leaderboard(object):
    '''
    A fixed capacity leading board data structure.

    It is a binary heap with fixed capacity, and automatically pop the elements
    with largest cost.
    '''
    def __init__(self, capacity, cost_func, dtype):

        self.capacity = capacity

        self.cost_func = cost_func

        self.data = np.empty(capacity + 1, dtype = dtype)

        self.size = 0

    def __heapInsert__(self, x):

        self.data[self.size] = x

        i = self.size

        while i > 0 and self.cost_func(self.data[i]) > self.cost_func(self.data[(i - 1) // 2]):

            (self.data[i], self.data[(i - 1) // 2]) = (self.data[(i - 1) // 2], self.data[i])

            i = (i - 1) // 2

        self.size += 1

    def __heapRemoveTop__(self):

        if self.size == 0:

            return

        elif self.size == 1:

            self.size = 0

        else:

            self.data[0] = self.data[self.size - 1]

            self.size -= 1

            i = 0

            while i * 2 + 1 < self.size:

                lft = i * 2 + 1
                rgt = lft + 1

                if rgt == self.size or self.cost_func(self.data[lft]) > self.cost_func(self.data[rgt]):
                    next_i = lft
                else:
                    next_i = rgt

                if self.cost_func(self.data[i]) > self.cost_func(self.data[next_i]):
                    break

                (self.data[i], self.data[next_i]) = (self.data[next_i], self.data[i])

                i = next_i

    def push(self, x):

        self.__heapInsert__(x)

        if self.size > self.capacity:
            self.pop()

    def pop(self):

        ret = self.data[0]

        self.__heapRemoveTop__()

        return ret

    def empty(self):

        return self.size == 0

if __name__ == '__main__':

    Q = Leaderboard(4, cost_func = (lambda x : x), dtype = int)

    Q.push(3)
    print(Q.size, Q.data)

    Q.push(1)
    print(Q.size, Q.data)

    Q.push(2)
    print(Q.size, Q.data)

    Q.push(5)
    print(Q.size, Q.data)

    Q.push(4)
    print(Q.size, Q.data)

    while not Q.empty():
        print(Q.pop(), end = '\0')
