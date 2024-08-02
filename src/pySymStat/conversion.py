__all__ = ['euler_to_quaternion',
           'quaternion_to_euler']

# src[0] -> AngleRot
# src[1] -> AngleTilt
# src[2] -> AnglePsi

import math
import numpy as np

from numpy.typing import NDArray

# src[0] -> AngleRot
# src[1] -> AngleTilt
# src[2] -> AnglePsi

def euler_to_quaternion(src : NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    The `euler_to_quaternion` function converts Euler angles (Relion's convention) to a unit quaternion.
    - `src`: This is a Numpy vector of type `np.float64` with a length of 3, representing `rlnAngleRot`, `rlnAngleTilt`, and `rlnAnglePsi` in Relion's starfile convention.
    '''
    psi = math.radians(src[0])
    theta = math.radians(src[1])
    phi = math.radians(src[2])

    w = math.cos((phi + psi) / 2) * math.cos(theta / 2)
    x = math.sin((phi - psi) / 2) * math.sin(theta / 2)
    y = math.cos((phi - psi) / 2) * math.sin(theta / 2)
    z = math.sin((phi + psi) / 2) * math.cos(theta / 2)

    return np.array([w, x, y, z], dtype = np.float64)

def quaternion_to_euler(src : NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    The `quaternion_to_euler` function converts a unit quaternion to Euler angles (Relion's convention).
    - `src`: This is a Numpy vector of type `np.float64` with a length of 4, which is a quaternion representing a spatial rotation.
    '''
    FLT_EPSILON = 1.19209*10**(-7)

    # mat : rotation matrix
    A = np.zeros((3,3))
    A[0,1] = -src[3]
    A[1,0] =  src[3]
    A[0,2] =  src[2]
    A[2,0] = -src[2]
    A[1,2] = -src[1]
    A[2,1] =  src[1]

    mat = np.eye(3) + 2 * src[0] * A + 2 * np.dot(A, A)
    mat = np.transpose(mat)

    if (abs(mat[1, 1]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 2] * mat[1, 2] * mat[2, 1] \
               - mat[0, 2] * mat[2, 0]) / mat[1, 1])
    elif (abs(mat[0, 1]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 1] * mat[2, 2] * mat[0, 2] + mat[2, 0] * mat[1, 2]) / mat[0, 1])
    elif (abs(mat[0, 0]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 0] * mat[2, 2] * mat[0, 2] - mat[2, 1] * mat[1, 2]) / mat[0, 0])
    else:
        print("Don't know how to extract angles.")
        exit()

    if (abs_sb > FLT_EPSILON):
        beta  = math.atan2(abs_sb, mat[2, 2])
        alpha = math.atan2(mat[2, 1] / abs_sb, mat[2, 0] / abs_sb)
        gamma = math.atan2(mat[1, 2] / abs_sb, -mat[0, 2] / abs_sb)
    else:
        alpha = 0
        beta  = 0
        gamma = math.atan2(mat[1, 0], mat[0, 0])

    gamma = math.degrees(gamma)
    beta  = math.degrees(beta)
    alpha = math.degrees(alpha)

    return np.array([alpha, beta, gamma], dtype = np.float64)
