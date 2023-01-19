__all__ = [
    'Leaderboard'
]

import numpy as np

class Leaderboard(object):
    '''
    A fixed capacity leader board data structure.

    It is a binary heap with fixed capacity, and automatically pop the elements
    with largest cost.
    '''
    def __init__(self, capacity, cost_func, dtype):
        self.capacity = capacity
        self.cost_func = cost_func
        self.data = np.empty(capacity + 1, dtype = dtype)
        self.size = 0

    def __heap_push__(self, x):
        self.data[self.size] = x
        i = self.size
        fi = (i - 1) // 2
        while i > 0 and self.cost_func(self.data[i]) > self.cost_func(self.data[fi]):
            self.data[i], self.data[fi] = self.data[fi], self.data[i]
            i = fi
            fi = (i - 1) // 2
        self.size += 1

    def __heap_pop__(self):
        if self.size == 0: return
        self.data[0] = self.data[self.size - 1]
        self.size -= 1
        i, lft, rgt = 0, 1, 2
        while lft < self.size:
            nxt = lft if rgt == self.size or self.cost_func(self.data[lft]) > self.cost_func(self.data[rgt]) else rgt
            if self.cost_func(self.data[i]) > self.cost_func(self.data[nxt]): return
            self.data[i], self.data[nxt] = self.data[nxt], self.data[i]
            i = nxt
            lft, rgt = i * 2 + 1, i * 2 + 2

    def push(self, x):
        self.__heap_push__(x)
        if self.size > self.capacity: self.pop()

    def pop(self):
        ret = self.data[0]
        self.__heap_pop__()
        return ret

    def empty(self):
        return self.size == 0

if __name__ == '__main__':
    Q = Leaderboard(4, cost_func = lambda x : x, dtype = int)

    for i in [3, 1, 2, 5, 4]:
        Q.push(i)
        print(Q.size, Q.data)

    while not Q.empty(): print(Q.pop(), end = ' ')
