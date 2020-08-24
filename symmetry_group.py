import numpy as np

from math import sin, cos 

from .symmetry_group_elems_CN_DN import *
from .symmetry_group_elems_T_O_I import *
from .symmetry_group_table import *
from .symmetry_group_irreducible_rep_CN_DN import *
from .symmetry_group_irreducible_rep_T_O_I import *

__all__ = ['get_sym_grp']

def get_sym_grp(sym):

    assert((sym[0] == 'C') or (sym[0] == 'D') or (sym == 'T') or (sym == 'O') or (sym == 'I'))

    if sym[0] == 'C':

        order = int(sym[1:])

        sym_grp_elems, _ = get_sym_grp_elems_CN(order)
        sym_grp_irreducible_rep = get_sym_grp_irreducible_rep_CN(order)
    
    elif sym[0] == 'D':

        order = int(sym[1:])

        sym_grp_elems, _ = get_sym_grp_elems_DN(order)
        sym_grp_irreducible_rep = get_sym_grp_irreducible_rep_DN(order)

    elif sym == 'T':

        sym_grp_elems, sym_grp_ab = get_sym_grp_elems_T()
        sym_grp_irreducible_rep = get_sym_grp_irreducible_rep_T(sym_grp_ab)

    elif sym == 'O':

        sym_grp_elems, sym_grp_ab = get_sym_grp_elems_O()
        sym_grp_irreducible_rep = get_sym_grp_irreducible_rep_O(sym_grp_ab)

    elif sym == 'I':

        sym_grp_elems, sym_grp_ab = get_sym_grp_elems_I()
        sym_grp_irreducible_rep = get_sym_grp_irreducible_rep_I(sym_grp_ab)

    sym_grp_table = get_sym_grp_table(sym_grp_elems)

    return sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep

if __name__ == '__main__':

    print(get_sym_grp('C1'))
    print(get_sym_grp('C3'))
    print(get_sym_grp('D3'))
    print(get_sym_grp('D4'))
    print(get_sym_grp('T'))
    print(get_sym_grp('O'))
    print(get_sym_grp('I'))
