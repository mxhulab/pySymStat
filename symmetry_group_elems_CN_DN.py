import numpy as np

from math import sin, cos 

__all__ = ['get_sym_grp_elems_CN', \
           'get_sym_grp_elems_DN']

def get_sym_grp_elems_CN(order):

    # sym_grp part

    sym_grp = np.zeros((order, 4), dtype = np.float64)

    sym_grp[:, 0] = np.array([cos(theta / 2) for theta in np.linspace(0, 2 * np.pi, order, endpoint = False)])
    sym_grp[:, 3] = np.array([sin(theta / 2) for theta in np.linspace(0, 2 * np.pi, order, endpoint = False)])

    # sym_grp_ab part

    sym_grp_ab = ['a' * i for i in range(order)]

    return sym_grp, sym_grp_ab

def get_sym_grp_elems_DN(order):

    # sym_grp part

    sym_grp = np.zeros((order * 2, 4), dtype = np.float64)

    sym_grp[:order] = get_sym_grp_elems_CN(order)[0]

    sym_grp[-order:, 1] = sym_grp[:order, 0]
    sym_grp[-order:, 2] = sym_grp[:order, 3]

    # sym_grp_ab part

    sym_grp_ab = ['a' * i for i in range(order)] + ['b' + 'a' * i for i in range(order)]

    return sym_grp, sym_grp_ab

if __name__ == '__main__':

    sym_grp, sym_grp_ab = get_sym_grp_elems_CN(5)

    print(sym_grp)
    print(sym_grp_ab)

    sym_grp, sym_grp_ab = get_sym_grp_elems_DN(5)

    print(sym_grp)
    print(sym_grp_ab)
