__all__ = [
    'mean_S2',
    'variance_S2',
    'mean_SO3',
    'variance_SO3',
    'mean_variance_S2_G',
    'mean_variance_SO3_G'
]

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Tuple, Callable, Union
from .distance import distance_S2, distance_SO3, action_S2, action_SO3
from .NUG import solve_NUG
from .symmetry import Symmetry

def mean_S2(
    vs : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> NDArray[np.float64]:
    '''Mean on S^2.

    Parameters
    ----------
    vs : array of shape (N, 3)
        Set of projection directions in unit vector form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    m : array of shape (3, )
        Mean projection direction.

    Raise
    -----
    NotImplementedError
        If the type of distance is geometric.
    '''
    assert vs.ndim == 2 and vs.shape[1] == 3
    if type == 'arithmetic':
        mean = np.mean(vs, axis = 0)
        return mean / np.linalg.norm(mean)
    elif type == 'geometric':
        raise NotImplementedError('Only arithmetic distance supported in mean_S2.')
    else:
        raise ValueError('Invalid type.')

def variance_S2(
    vs : NDArray[np.float64],
    type : Literal['arithmetic'] = 'arithmetic',
    mean : Optional[NDArray[np.float64]] = None
) -> float:
    '''Variance on S^2.

    Parameters
    ----------
    vs : array of shape (N, 3)
        Set of projection directions in unit vector form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.
    mean : array of shape (3, ), optional
        Reference mean vector, default is None.
        If this is given, then the variance is computed as 1/N\sum_i d(vs[i], mean)^2.
        Otherwise, the variance is 1/2N^2\sum_{i,j}d(vs[i], vs[j])^2.

    Returns
    -------
    d : float
        Variance.
    '''
    assert vs.ndim == 2 and vs.shape[1] == 3
    if mean is None:
        dist = distance_S2(vs.T[:, :, np.newaxis], vs.T[:, np.newaxis, :], type)
        return np.mean(np.square(dist)) / 2
    else:
        assert mean.shape == (3, )
        dist = distance_S2(vs.T, mean[:, None], type)
        return np.mean(np.square(dist))

def mean_SO3(
    qs : NDArray[np.float64],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic'
) -> NDArray[np.float64]:
    '''Mean on SO(3).

    Parameters
    ----------
    qs : array of shape (N, 4)
        Set of spatial rotations in unit quaternion form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    m : array of shape (4, )
        Mean spatial rotation.

    Raise
    -----
    NotImplementedError
        If the type of distance is geometric.
    '''
    assert qs.ndim == 2 and qs.shape[1] == 4
    if type == 'arithmetic':
        t = np.tensordot(qs, qs, axes = (0, 0))
        _, eigvects = np.linalg.eigh(t)
        return eigvects[:, -1].copy()
    elif type == 'geometric':
        raise NotImplementedError('Only arithmetic distance supported in mean_SO3.')
    else:
        raise ValueError('Invalid type.')

def variance_SO3(
    qs : NDArray[np.float64],
    type : Literal['arithmetic'] = 'arithmetic',
    mean : Optional[NDArray[np.float64]] = None
) -> float:
    '''Variance on SO(3).

    Parameters
    ----------
    qs : array of shape (N, 4)
        Set of spatial rotations in unit quaternion form.
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.
    mean : array of shape (4, ), optional
        Reference mean quaternion, default is None.
        If this is given, then the variance is computed as 1/N\sum_i d(qs[i], mean)^2.
        Otherwise, the variance is 1/2N^2\sum_{i,j}d(qs[i], qs[j])^2.

    Returns
    -------
    d : float
        Variance.
    '''
    assert qs.ndim == 2 and qs.shape[1] == 4
    if mean is None:
        dist = distance_SO3(qs.T[:, :, np.newaxis], qs.T[:, np.newaxis, :], type)
        return np.mean(np.square(dist)) / 2
    else:
        assert mean.shape == (4, )
        dist = distance_SO3(qs.T, mean[:, None], type)
        return np.mean(np.square(dist))

def _mean_variance_M_G(
    data : NDArray[np.float64],
    sym : Symmetry,
    type : str,
    action : Callable,
    distance : Callable,
    mean_func : Callable,
    variance_func : Callable,
    **kwargs
) -> Tuple[NDArray[np.float64], float, float, NDArray[np.int32], NDArray[np.float64]]:

    # Construct and solve NUG.
    f = distance(action(data.T, sym.elems.T, pairwise = True), data.T, type = type, pairwise = True)
    f = np.square(np.transpose(f, axes = (1, 0, 2)))
    solutions = solve_NUG(f, sym, **kwargs)
    representatives = action(data.T, sym.elems[solutions].T).T
    mean = variance = None
    try:
        mean = mean_func(representatives, type = type)
        variance = variance_func(representatives, type = type, mean = mean)
    except:
        pass
    cost = variance_func(representatives, type = type)
    return mean, variance, cost, solutions, representatives

def mean_variance_SO3_G(
    qs : NDArray[np.float64],
    sym : Union[str, Symmetry],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic',
    **kwargs
) -> Tuple[Optional[NDArray[np.float64]], Optional[float], float, NDArray[np.int32], NDArray[np.float64]]:
    '''Mean and variance on SO(3)/G.

    Parameters
    ----------
    qs : array of shape (N, 4)
        Set of spatial rotations in unit quaternion form.
    sym : molecular symmetry group
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    mean : array of shape (4, ), optional
        Mean spatial rotation.
        For geometric distance, returns None.
    variance : float, optional
        Variance.
        For geometric distance, returns None.
    cost : float
        The optimal cost in NUG problem.
        It is equal to the reference-free version variance.
    solutions : array of shape (N, )
        Optimal solutions to clustering those vectors, i.e., optimal g[i].
    representatives : array of shape (N, 4)
        Representatives, i.e., q[i]g[i].
    '''
    assert qs.ndim == 2 and qs.shape[1] == 4
    if isinstance(sym, str): sym = Symmetry(sym)
    assert isinstance(sym, Symmetry)
    return _mean_variance_M_G(qs, sym, type, action_SO3, distance_SO3, mean_SO3, variance_SO3, **kwargs)

def mean_variance_S2_G(
    vs : NDArray[np.float64],
    sym : Union[str, Symmetry],
    type : Literal['arithmetic', 'geometric'] = 'arithmetic',
    **kwargs
) -> Tuple[Optional[NDArray[np.float64]], Optional[float], float, NDArray[np.int32], NDArray[np.float64]]:
    '''Mean and variance on S^2/G.

    Parameters
    ----------
    vs : array of shape (N, 3)
        Set of projection directions in unit vector form.
    sym : molecular symmetry group
    type : 'arithmetic' or 'geometric'
        Type of distance, default is 'arithmetic'.

    Returns
    -------
    mean : array of shape (3, ), optional
        Mean projection direction.
        For geometric distance, returns None.
    variance : float, optional
        Variance.
        For geometric distance, returns None.
    cost : float
        The optimal cost in NUG problem.
        It is equal to the reference-free version variance.
    solutions : array of shape (N, )
        Optimal solutions to clustering those vectors, i.e., optimal g[i].
    representatives : array of shape (N, 3)
        Representatives, i.e., rotate(g[i])^Tv[i].
    '''
    assert vs.ndim == 2 and vs.shape[1] == 3
    if isinstance(sym, str): sym = Symmetry(sym)
    assert isinstance(sym, Symmetry)
    return _mean_variance_M_G(vs, sym, type, action_S2, distance_S2, mean_S2, variance_S2, **kwargs)

if __name__ == '__main__':
    vs = np.random.randn(5, 3)
    vs /= np.linalg.norm(vs, axis = 1, keepdims = True)
    m = mean_S2(vs, 'arithmetic')
    f = lambda x: x - x ** 2 / 4
    var1 = variance_S2(vs, 'arithmetic')
    var2 = variance_S2(vs, 'arithmetic', m)
    assert np.allclose(f(var2), var1)

    print('vs =', vs)
    print('Mean^A[vs] =', m)
    print('Var^A[vs] =', var1)
    print('Mean-based Var^A[vs] =', var2)
    print()

    print('Mean^G[vs] =', NotImplemented)
    print('Var^G[vs] =', variance_S2(vs, 'geometric'))
    print('Arithemetic mean-based Var^A[vs] =', variance_S2(vs, 'geometric', m))
    print()

    qs = np.random.randn(10, 4)
    qs /= np.linalg.norm(qs, axis = 1, keepdims = True)
    m = mean_SO3(qs, 'arithmetic')
    f1 = lambda x: x - x ** 2 / 8 if x <= 4 else \
                   -8 + 4 * x - 3 / 8 * x ** 2 if x <= 16 / 3 else \
                   -24 + 9 * x - 3 / 4 * x ** 2
    f2 = lambda x: x - x ** 2 / 12
    var1 = variance_SO3(qs, 'arithmetic')
    var2 = variance_SO3(qs, 'arithmetic', m)
    assert f1(var2) <= var1 <= f2(var2)

    print('qs =', qs)
    print('Mean^A[qs] =', m)
    print('Var^A[qs] =', var1)
    print('Mean-based Var^A[vs] =', var2)
    print()
