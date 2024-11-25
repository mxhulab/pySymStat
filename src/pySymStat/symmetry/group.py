import re
import sys
import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray
from .cyclic import get_grp_info as C_info
from .dihedral import get_grp_info as D_info
from .tetrahedral import get_grp_info as T_info
from .octahedral import get_grp_info as O_info
from .icosahedral import get_grp_info as I_info
from .table import get_g_table, get_i_table

def _parse_grp_name(sym : str) -> Tuple[str, int]:
    '''Check and parse name of a symmetry groups.

    Parameters
    ----------
    sym : name of the symmetry group

    Returns
    -------
    grp : name of the group, must be one of ['C', 'D', 'T', 'O', 'I']
    rank: n

    Raise
    -----
    ValueError
        If the group name is invalid.
    '''
    pattern = r'^(C([1-9][0-9]*)|D([1-9][0-9]*)|T|O|I|I([1-4]))$'
    match = re.match(pattern, sym, re.IGNORECASE)
    if not match:
        raise ValueError(f'Invalid group name {sym}.')

    grp = match.group(1)[0].upper()
    if grp == 'C':
        return grp, int(match.group(2))
    elif grp == 'D':
        return grp, int(match.group(3))
    elif grp in ['T', 'O']:
        return grp, 1
    else:
        return grp, 2 if match.group(4) is None else int(match.group(4))

class Symmetry(object):
    '''Molecular symmetry group class.
    '''
    def __init__(self, sym):
        '''Construct a molecular symmetry group object.

        This object records the group elements (in unit quaternion form), group table,
        inverse table, real irreducible representations and their degree of splitting,
        of the molecular symmetry group.

        Supported name for molecular symmetry groups:
        - Cn (or cn): cyclic group of order n.
        - Dn (or dn): dihedral group of order 2n.
        - T: tetrahedral group.
        - O: octahedral group.
        - I1 | I2(I) | I3 | I4: icosahedral group.

        There are different conventions for icosahedral groups.
        We follow the convention in RELION, which supports 4 different conventions.
        The default one is I2.

        For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).

        Parameters
        ----------
        sym : name of the symmetry group

        Raise
        -----
        ValueError
            If the group name is invalid.
        '''
        if isinstance(sym, Symmetry):
            self._name = sym._name
            self._elems = sym._elems
            self._g_table = sym._g_table
            self._i_table = sym._i_table
            self._irreps = sym._irreps
            self._splits = sym._splits
        else:
            grp, rank = _parse_grp_name(sym)
            self._name = f'{grp}{rank}'
            func_dict = {
                'C' : C_info,
                'D' : D_info,
                'T' : T_info,
                'O' : O_info,
                'I' : I_info
            }
            self._elems, self._irreps, self._splits = func_dict[grp](rank)
            self._g_table = get_g_table(self._elems)
            self._i_table = get_i_table(self._g_table)

    @property
    def size(self):
        return len(self.g_table)

    @property
    def name(self):
        if self._name == 'O1':
            return 'O'
        elif self._name == 'T1':
            return 'T'
        elif self._name == 'I2':
            return 'I'
        else:
            return self._name

    @property
    def elems(self) -> NDArray[np.float64]:
        '''elems : array of shape (M, 4)

        M is the order/size of the group.
        '''
        return self._elems

    @property
    def g_table(self) -> NDArray[np.int32]:
        '''g_table : array of shape (M, M)

        The group table, satisfying elems[i] * elems[j] == elems[g_table[i, j]].
        '''
        return self._g_table

    @property
    def i_table(self) -> NDArray[np.int32]:
        '''i_table : array of shape (M, M)

        The inverse table, satisfying elems[i] * elems[i_table[i]] == elems[0].'''
        return self._i_table

    @property
    def irreps(self) -> List[NDArray[np.float64]]:
        '''irreps : List of real irreps

        `irreps[k]` is the k-th irrep of shape = (M, dk, dk),
        where dk is the dimension of k-th real irrep.
        '''
        return self._irreps

    @property
    def splits(self) -> List[int]:
        '''s_irreps : List of int

        `s_irreps[k]` is the degree of splitting of k-th real irrep.
        '''
        return self._splits

    def save(self, fpath : str):
        '''Save all informations of a molecular symmetry group into a file.
        '''
        with open(fpath, 'w') as fout:
            self.print(fout)

    def print(self, fout = sys.stdout):
        '''Print all informations of a molecular symmetry group.
        '''
        M, K = len(self._g_table), len(self._irreps)
        print(M, file = fout)

        formatter = {'int' : lambda x: f'{x:3d}', 'float' : lambda x: f'{x:12.8f}'}
        config = {
            'formatter' : formatter,
            'separator' : ' ',
            'max_line_width' : np.inf,
            'threshold' : np.inf
        }

        assert self._elems.shape == (M, 4) and self._elems.dtype == np.float64
        print(np.array2string(self._elems, **config), file = fout)
        print(file = fout)

        assert self._g_table.shape == (M, M) and self._g_table.dtype == np.int32
        print(np.array2string(self._g_table, **config), file = fout)
        print(file = fout)

        assert self._i_table.shape == (M, ) and self._i_table.dtype == np.int32
        print(np.array2string(self._i_table, **config), file = fout)
        print(file = fout)

        for k in range(K):
            dk, sk = self._irreps[k].shape[1], self._splits[k]
            print(dk, sk, file = fout)
            assert self._irreps[k].shape == (M, dk, dk) and self._irreps[k].dtype == np.float64
            print(np.array2string(self._irreps[k], **config), file = fout)

    def verify(self) -> bool:
        '''Quickly verify the informations of a molecular symmetry group.
        '''
        M, K = len(self._g_table), len(self._irreps)

        print('Check whether the elements are in unit quaternions.')
        assert self._elems.shape == (M, 4) and self._elems.dtype == np.float64
        assert np.allclose(np.linalg.norm(self._elems, axis = 1), 1)

        print('Check wheter the multiplication table of group is correct.')
        from .generate import _quat_mult, _quat_same
        assert self._g_table.shape == (M, M) and self._g_table.dtype == np.int32
        for g1, g2 in np.ndindex(M, M):
            assert _quat_same(_quat_mult(self._elems[g1], self._elems[g2]), self._elems[self._g_table[g1, g2]])

        print('Check wheter the inverse table of group is correct.')
        assert self._i_table.shape == (M, ) and self._i_table.dtype == np.int32
        for g in range(M):
            assert self._g_table[g, self._i_table[g]] == 0 and self._g_table[self._i_table[g], g] == 0

        for k in range(K):
            print(f'Check whether the irrep {k} is correct.')
            dk = self._irreps[k].shape[1]
            Ik = np.eye(dk, dtype = np.float64)
            rhok = self._irreps[k]
            assert rhok.shape == (M, dk, dk) and rhok.dtype == np.float64
            for g in range(M):
                assert np.linalg.norm(rhok[g] @ rhok[g].T - Ik) < 1e-6
            for g1, g2 in np.ndindex(M, M):
                assert np.linalg.norm(rhok[g1] @ rhok[g2] - rhok[self._g_table[g1, g2]]) < 1e-6
