import numpy as np
from numpy.typing import NDArray

def quat_mult(q1 : NDArray[np.float64], q2 : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Product of two quaternions.

    Parameters
    ----------
    q1 : array of shape (4, ...)
    q1 : array of shape (4, ...)

    Returns
    -------
    q : q1 * q2
    '''
    assert q1.shape[0] == q2.shape[0] == 4
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    ], dtype = np.float64)

def quat_norm(q : NDArray[np.float64]) -> float:
    '''Norm of a quaternion.
    '''
    assert q.shape[0] == 4
    return np.linalg.norm(q, axis = 0)

def quat_norm2(q : NDArray[np.float64]) -> np.float64:
    '''Square norm of a quaternion.
    '''
    assert q.shape[0] == 4
    return np.sum(q ** 2, axis = 0)

def quat_conj(q : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Conjugation of a quaternion.
    '''
    assert q.shape[0] == 4
    q_conj = q.copy()
    q_conj[1:] *= -1
    return q_conj

def quat_inv(q : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Inverse of a quaternion.
    '''
    assert q.shape[0] == 4
    return quat_conj(q) / quat_norm2(q)

def quat_rotate(q : NDArray[np.float64], v : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Rotate a vector by a quaternion.

    Parameters
    ----------
    q : array of shape (4, ...)
    v : array of shape (3, ...)

    Returns
    -------
    Rotate(q)v : array of shape (3, ...)
    '''
    assert q.shape[0] == 4 and v.shape[0] == 3
    w = np.empty((4, *v.shape[1:]), dtype = np.float64)
    w[0], w[1:] = 0, v
    return quat_mult(quat_mult(q, w), quat_inv(q))[1:]

if __name__ == '__main__':
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    v = np.random.randn(3)
    v /= np.linalg.norm(v)

    print('q1 =', q1)
    print('q2 =', q2)
    print('q1 * q2 =', quat_mult(q1, q2))
    print('norm(q1) =', quat_norm(q1))
    print('conj(q1) =', quat_conj(q1))
    print('inv(q1) =', quat_inv(q1))

    v1 = quat_rotate(q1, v)
    v2 = quat_rotate(-q1, v)
    v3 = quat_rotate(quat_conj(q1), v1)
    print('rotate(q1)v =', v1)
    print('rotate(-q1)v =', v2)
    assert np.allclose(v1, v2)
    assert np.allclose(v, v3)

    v4 = quat_rotate(q2, v1)
    v5 = quat_rotate(quat_mult(q2, q1), v)
    print('rotate(q2)rotate(q1)v =', v4)
    print('rotate(q2 * q1)v =', v5)
    assert np.allclose(v4, v5)
