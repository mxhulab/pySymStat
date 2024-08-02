__all__ = [
    'quat_norm', # hide this
    'quat_conj',
    'quat_mult',
    'quat_same', # hide this
    'quat_rotate'
]

import numpy as np

from numpy.typing import NDArray

def quat_norm(q : NDArray[np.float64]) -> float:
    return np.linalg.norm(q)

def quat_conj(q : NDArray[np.float64]) -> NDArray[np.float64]:
    return q * np.array([1, -1, -1, -1])

def quat_mult(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64]
) -> NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype = np.float64)

def quat_same(
    q1 : NDArray[np.float64],
    q2 : NDArray[np.float64],
    eps : float = 1e-3
) -> bool:
    return np.linalg.norm(q1 - q2) < eps or np.linalg.norm(q1 + q2) < eps

def quat_rotate(
    q : NDArray[np.float64], 
    v : NDArray[np.float64]
) -> NDArray[np.float64]:
    v = np.array([0, v[0], v[1], v[2]], dtype = np.float64)
    qv = quat_mult(quat_mult(q, v), quat_conj(q))
    return qv[1:]

if __name__ == '__main__':

    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    v = np.random.randn(3)
    v /= np.linalg.norm(v)

    print(r'R_q v == R_{-q} v:',
          np.allclose(quat_rotate(q1, v), quat_rotate(-q1, v)))
    print(r'R_{conj(q)} R_q v == v:',
          np.allclose(quat_rotate(quat_conj(q1), quat_rotate(q1, v)), v))
    print(r'R_{q1} R_{q2} v == R_{q1 q2}v:',
          np.allclose(quat_rotate(q1, quat_rotate(q2, v)), quat_rotate(quat_mult(q1, q2), v)))
