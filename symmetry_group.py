__all__ = [
    'get_sym_grp'
]

import numpy as np
from .quaternion import *
from .symmetry_group_Cn import get_sym_grp_Cn_elems, get_sym_grp_Cn_irreps
from .symmetry_group_Dn import get_sym_grp_Dn_elems, get_sym_grp_Dn_irreps
from .symmetry_group_TOI import get_sym_grp_TOI_elems, get_sym_grp_TOI_irreps

def get_sym_grp(sym : str):
    '''
    parameters
    ==========
    sym : str
        'Cn' | 'Dn' | 'T' | 'O' | 'I'

    returns
    =======
    (sym_grp_elems, sym_grp_table, sym_grp_irreps)
        See doc for details.
    '''
    if sym[0] == 'C':
        n = int(sym[1:])
        sym_grp_elems = get_sym_grp_Cn_elems(n)
        sym_grp_irreps = get_sym_grp_Cn_irreps(n)

    elif sym[0] == 'D':
        n = int(sym[1:])
        sym_grp_elems = get_sym_grp_Dn_elems(n)
        sym_grp_irreps = get_sym_grp_Dn_irreps(n)

    elif sym[0] in ['T', 'O', 'I']:
        sym_grp_elems, sym_grp_elems_ab = get_sym_grp_TOI_elems(sym[0])
        sym_grp_irreps = get_sym_grp_TOI_irreps(sym[0], sym_grp_elems_ab)

    else:
        raise ValueError('Invalid argument.')

    # Construct the group table.
    n = len(sym_grp_elems)
    sym_grp_table = np.empty((n, n), dtype = np.int32)
    for i1, i2 in np.ndindex(n, n):
        q1 = sym_grp_elems[i1]
        q2 = sym_grp_elems[i2]
        q = quat_mult(q1, q2)
        for i in range(n):
            if quat_same(q, sym_grp_elems[i]):
                sym_grp_table[i1, i2] = i
                break

    return sym_grp_elems, sym_grp_table, sym_grp_irreps

def pretty_print(sym_grp_info):
    sym_grp_elems, sym_grp_table, sym_grp_irreps = sym_grp_info
    n = len(sym_grp_elems)
    k = len(sym_grp_irreps)

    print(n, 'elems in this grp:')
    for q in sym_grp_elems:
        print(f'  [{q[0]:11.8f} {q[1]:11.8f} {q[2]:11.8f} {q[3]:11.8f}]')

    print('Group table:')
    for i1 in range(n):
        print('  ', end = '')
        for i2 in range(n):
            print(f'{sym_grp_table[i1, i2]:2d} ', end = '')
        print()

    print(k, 'irreps of this grp:')
    for i in range(k):
        dim = sym_grp_irreps[i, 0].shape[0]
        is_real = sym_grp_irreps[i, 0].dtype == np.float64
        print(f'  Irrep {i}, dimension {dim},', 'real.' if is_real else 'complex.')

        for j in range(n):
            for l1 in range(dim):
                print('    ', end = '')
                for l2 in range(dim):
                    if is_real:
                        print(f'{sym_grp_irreps[i, j][l1, l2]:11.8f} ', end = '')
                    else:
                        print(f'{sym_grp_irreps[i, j][l1, l2]:23.8f} ', end = '')
                print()

if __name__ == '__main__':
    pretty_print(get_sym_grp('C1'))
    pretty_print(get_sym_grp('C3'))
    pretty_print(get_sym_grp('C4'))
    pretty_print(get_sym_grp('D1'))
    pretty_print(get_sym_grp('D3'))
    pretty_print(get_sym_grp('D4'))
    pretty_print(get_sym_grp('T'))
    pretty_print(get_sym_grp('O'))
    pretty_print(get_sym_grp('I'))
