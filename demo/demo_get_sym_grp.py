import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    for group in ['C2', 'C7', 'D2', 'D7', 'T', 'O', 'I1', 'I2', 'I3']:

        sym_grp_elems, sym_grp_table, sym_grp_irrep = pySymStat.get_sym_grp(group)

        print("================================================================================")
        print("================================================================================")
        print("Group elements, Caley table and irreducible representation of symmetry group {}.".format(group))
        print("================================================================================")
        print("Group elements: \n", sym_grp_elems)
        print("Cayley table: \n", sym_grp_table)
        print("Irreducible representation: \n", sym_grp_irrep)
        print("")
