import numpy as np
from numpy.typing import NDArray

def euler_to_quat(angle : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Convert spatial rotation in Euler angles form to unit quaternion form.

    For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#orientations).

    Parameters
    ----------
    angle : array of shape (3, ...)
        Spatial rotations in Euler angles form.

    Returns
    -------
    q : array of shape (4, ...)
        Spatial rotations in unit quaternion form.
    '''
    assert angle.shape[0] == 3
    q = np.empty((4, *angle.shape[1:]), dtype = np.float64)
    angle = np.radians(angle)
    q[0] =  np.cos((angle[2] + angle[0]) / 2) * np.cos(angle[1] / 2)
    q[1] = -np.sin((angle[2] - angle[0]) / 2) * np.sin(angle[1] / 2)
    q[2] = -np.cos((angle[2] - angle[0]) / 2) * np.sin(angle[1] / 2)
    q[3] = -np.sin((angle[2] + angle[0]) / 2) * np.cos(angle[1] / 2)
    return q

def quat_to_euler(q : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Convert spatial rotation in unit quaternion form to Euler angles form.

    For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#orientations).

    Parameters
    ----------
    q : array of shape (4, ...)
        Spatial rotations in unit quaternion form.

    Returns
    -------
    angle : array of shape (3, ...)
        Spatial rotations in Euler angles form.
    '''
    assert q.shape[0] == 4
    angle = np.empty((3, *q.shape[1:]), dtype = np.float64)
    angle[1] = np.arctan2(np.hypot(q[1], q[2]), np.hypot(q[0], q[3])) * 2
    s = np.arctan2(-q[3],  q[0])
    m = np.arctan2(-q[1], -q[2])
    angle[0] = s - m
    angle[2] = s + m
    angle[0] = np.remainder(angle[0] + np.pi, 2 * np.pi) - np.pi
    angle[2] = np.remainder(angle[2] + np.pi, 2 * np.pi) - np.pi
    return np.degrees(angle)

def euler_to_vec(angle : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Convert projection direction in Euler angles form to unit vector form.

    For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#orientations).

    Parameters
    ----------
    angle : array of shape (2, ...)
        Projection directions in Euler angles form.

    Returns
    -------
    v : array of shape (3, ...)
        Projection directions in unit vector form.
    '''
    assert angle.shape[0] == 2
    v = np.empty((3, *angle.shape[1:]), dtype = np.float64)
    angle = np.radians(angle)
    v[0] = np.sin(angle[1]) * np.cos(angle[0])
    v[1] = np.sin(angle[1]) * np.sin(angle[0])
    v[2] = np.cos(angle[1])
    return v

def vec_to_euler(v : NDArray[np.float64]) -> NDArray[np.float64]:
    '''Convert spatial rotation in unit vector form to Euler angles form.

    For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#orientations).

    Parameters
    ----------
    v : array of shape (3, ...)
        Spatial rotations in unit quaternion form.

    Returns
    -------
    angle : array of shape (2, ...)
        Spatial rotations in Euler angles form.
    '''
    assert v.shape[0] == 3
    angle = np.empty((2, *v.shape[1:]), dtype = np.float64)
    angle[0] = np.arctan2(v[1], v[0])
    angle[1] = np.arctan2(np.hypot(v[0], v[1]), v[2])
    return np.degrees(angle)
