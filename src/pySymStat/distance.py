__all__ = [
    'distance_SO3',
    'distance_S2',
    'distance_SO3_G',
    'distance_S2_G',
    'action_S2',
    'action_SO3'
]

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from .quaternion import *
from .symmetry import Symmetry

def _stable_acos(x : NDArray[np.float64]) -> NDArray[np.float64]:
    return np.arccos(np.fmax(-1, np.fmin(1, x)))

def _stable_sqrt(x : NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(np.fmax(0, x))

def distance_SO3(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic',
    pairwise : bool = False
) -> NDArray[np.float64]:
    '''Arithmetic or geometric distance on SO(3).

    Parameters
    ----------
    q1 : array of shape (4, ...)
        Spatial rotation in unit quaternion form.
    q2 : array of shape (4, ...)
        Spatial rotation in unit quaternion form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.
    pairwise : bool
        Calculate distance in pairwise mode instead of broadcasting mode.
        Distance between quaternions in shape (4, *n1) and (4, *n2) will be of shape (*n1, *n2).

    Returns
    -------
    d : array of shape (...)
        Distance between q1 and q2.
    '''
    assert q1.shape[0] == q2.shape[0] == 4
    if type == 'arithmetic':
        if pairwise:
            return np.sqrt(8) * _stable_sqrt(1 - np.square(np.tensordot(q1, q2, axes = (0, 0))))
        else:
            return np.sqrt(8) * _stable_sqrt(1 - np.square(np.sum(q1 * q2, axis = 0)))
    elif type == 'geometric':
        if pairwise:
            return 2 * _stable_acos(np.abs(np.tensordot(q1, q2, axes = (0, 0))))
        else:
            return 2 * _stable_acos(np.abs(np.sum(q1 * q2, axis = 0)))
    else:
        raise ValueError('Invalid type.')

def distance_S2(
    v1 : NDArray[np.float64],
    v2 : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic',
    pairwise : bool = False
) -> NDArray[np.float64]:
    '''Arithmetic or geometric distance on S2.

    Parameters
    ----------
    v1 : array of shape (3, ...)
        Projection direction in unit vector form.
    v2 : array of shape (3, ...)
        Projection direction in unit vector form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.
    pairwise : bool
        Calculate distance in pairwise mode instead of broadcasting mode.
        Distance between quaternions in shape (3, *n1) and (3, *n2) will be of shape (*n1, *n2).

    Returns
    -------
    d : array of shape (...)
        Distance between v1 and v2.
    '''
    assert v1.shape[0] == v2.shape[0] == 3
    if type == 'arithmetic':
        if pairwise:
            newshape1 = (3, *v1.shape[1:], *map(lambda _ : 1, v2.shape[1:]))
            newshape2 = (3, *map(lambda _ : 1, v1.shape[1:]), *v2.shape[1:])
            return np.linalg.norm(v1.reshape(newshape1) - v2.reshape(newshape2), axis = 0)
        else:
            return np.linalg.norm(v1 - v2, axis = 0)
    elif type == 'geometric':
        if pairwise:
            return _stable_acos(np.tensordot(v1, v2, axes = (0, 0)))
        else:
            return _stable_acos(np.sum(v1 * v2, axis = 0))
    else:
        raise ValueError('Invalid type.')

def action_SO3(
    q : NDArray[np.float64],
    qg : NDArray[np.float64],
    pairwise : bool = False
) -> NDArray[np.float64]:
    '''Group action (on the right) of SO(3) on SO(3).

    Parameters
    ----------
    q : array of shape (4, ...)
        Spatial rotation in unit quaternion form.
    qg : array of shape (4, ...)
        Group element of G in unit quaternion form.
    pairwise : bool
        Calculate the action in pairwise mode instead of broadcasting mode.
        The results of action of q in shape (4, *n1) by qg in shape(4, *n2)
        will be of shape (4, *n1, *n2).

    Returns
    -------
    q * qg : array of shape (4, ...)
    '''
    assert q.shape[0] == qg.shape[0] == 4
    if pairwise:
        newshape1 = (4, *q.shape[1:], *map(lambda _ : 1, qg.shape[1:]))
        newshape2 = (4, *map(lambda _ : 1, q.shape[1:]), *qg.shape[1:])
        return quat_mult(q.reshape(newshape1), qg.reshape(newshape2))
    else:
        return quat_mult(q, qg)

def action_S2(
    v : NDArray[np.float64],
    qg : NDArray[np.float64],
    pairwise : bool = False
) -> NDArray[np.float64]:
    '''Group action (on the right) of SO(3) on S^2.

    Parameters
    ----------
    v : array of shape (3, ...)
        Projection direction in unit vector form.
    qg : array of shape (4, ...)
        Group element of G in unit quaternion form.
    pairwise : bool
        Calculate the action in pairwise mode instead of broadcasting mode.
        The results of action of v in shape (3, *n1) by qg in shape(4, *n2)
        will be of shape (3, *n1, *n2).

    Returns
    -------
    Rotate(qg^{-1})v : array of shape (3, ...)
    '''
    assert v.shape[0] == 3 and qg.shape[0] == 4
    if pairwise:
        newshape1 = (3, *v.shape[1:], *map(lambda _ : 1, qg.shape[1:]))
        newshape2 = (4, *map(lambda _ : 1, v.shape[1:]), *qg.shape[1:])
        return quat_rotate(quat_conj(qg.reshape(newshape2)), v.reshape(newshape1))
    else:
        return quat_rotate(quat_conj(qg), v)

def distance_SO3_G(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64],
    sym : Union[str, Symmetry],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    '''Arithmetic or geometric distance on SO(3)/G.

    Parameters
    ----------
    q1 : array of shape (4, ...)
        Spatial rotation in unit quaternion form.
    q2 : array of shape (4, ...)
        Spatial rotation in unit quaternion form.
    sym : str | Symmetry
        The molecular symmetry group.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    d : array of shape (...)
        Distance between [q1] and [q2] on quotient manifold SO(3)/G.
    g : array of shape (...)
        Index of the optimal group element such that d(q1, q2 * elems[g]) is minimal.
    qg : array of shape (4, ...)
        Optimal group element, i.e., elems[g].
    q2qg : array of shape (4, ...)
        Optimal representative, i.e., q2 * qg.
    '''
    assert q1.shape[0] and q2.shape[0] == 4
    if isinstance(sym, str): sym = Symmetry(sym)
    assert isinstance(sym, Symmetry)
    elems = sym.elems
    q2qg = action_SO3(q2, elems.T, pairwise = True)
    dist = distance_SO3(q1[..., None], q2qg, type)
    idx = np.argmin(dist, axis = -1)
    dist = np.amin(dist, axis = -1)
    qg = elems.T.take(idx, axis = -1)
    q2qg = np.take_along_axis(q2qg, idx[None, ..., None], axis = -1).squeeze(axis = -1)
    return dist, idx, qg, q2qg

def distance_S2_G(
    v1 : NDArray[np.float64],
    v2 : NDArray[np.float64],
    sym : Union[str, Symmetry],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    '''Arithmetic or geometric distance on S^2/G.

    Parameters
    ----------
    v1 : array of shape (3, ...)
        Projection direction in unit vector form.
    v2 : array of shape (3, ...)
        Projection direction in unit vector form.
    sym : str | Symmetry
        The molecular symmetry group.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    d : array of shape (...)
        Distance between [v1] and [v2] on quotient manifold S^2/G.
    g : array of shape (...)
        Index of the optimal group element such that d(v1, Rotate(elems[g]^{-1})v2) is minimal.
    qg : array of shape (4, ...)
        Optimal group element, i.e., elems[g].
    v2qg : array of shape (3, ...)
        Optimal representative, i.e., Rotate(elems[g]^{-1})v2.
    '''
    assert v1.shape[0] == v2.shape[0] == 3
    if isinstance(sym, str): sym = Symmetry(sym)
    assert isinstance(sym, Symmetry)
    v2qg = action_S2(v2, sym.elems.T, pairwise = True)
    dist = distance_S2(v1[..., None], v2qg, type)
    idx = np.argmin(dist, axis = -1)
    dist = np.amin(dist, axis = -1)
    qg = sym.elems.T.take(idx, axis = -1)
    v2qg = np.take_along_axis(v2qg, idx[None, ..., None], axis = -1).squeeze(axis = -1)
    return dist, idx, qg, v2qg

if __name__ == '__main__':
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(3)
    v2 /= np.linalg.norm(v2)

    print('q1 =', q1)
    print('q2 =', q2)
    print('d^A(q1, q2) =', distance_SO3(q1, q2, 'arithmetic'))
    print('d^G(q1, q2) =', distance_SO3(q1, q2, 'geometric'))
    print()

    print('v1 =', v1)
    print('v2 =', v2)
    print('d^A(v1, v2) =', distance_S2(v1, v2, 'arithmetic'))
    print('d^G(v1, v2) =', distance_S2(v1, v2, 'geometric'))
    print()

    from .symmetry import Symmetry
    sym = Symmetry('I')

    dist, g, qg, q2qg = distance_SO3_G(q1, q2, sym, 'arithmetic')
    print('d_{SO(3)/G}}^A(q1, q2) =', dist)
    print('  (g, qg, q2qg) =', g, qg, q2qg)
    dist, g, qg, q2qg = distance_SO3_G(q1, q2, sym, 'geometric')
    print('d_{SO(3)/G}^G(q1, q2) =', dist)
    print('  (g, qg, q2qg) =', g, qg, q2qg)
    print()

    dist, g, qg, v2qg = distance_S2_G(v1, v2, sym, 'arithmetic')
    print('d_{S^2/G}^A(v1, v2) =', dist)
    print('  (g, qg, v2qg) =', g, qg, v2qg)
    dist, g, qg, v2qg = distance_S2_G(v1, v2, sym, 'geometric')
    print('d_{S^2/G}^G(v1, v2) =', dist)
    print('  (g, qg, v2qg) =', g, qg, v2qg)
