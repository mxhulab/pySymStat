import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np

import pySymStat

if __name__ == '__main__':

    N = 5

    print("======================================================")
    print("Generate a 2D array, each row is a three-Euler-angles.")

    rot  = np.random.uniform(low=-180.0, high=180.0, size=N)
    tilt = np.random.uniform(low=   0.0, high=180.0, size=N)
    psi  = np.random.uniform(low=   0.0, high=360.0, size=N)

    relion_euler_angles = np.column_stack((rot, tilt, psi))
    print(relion_euler_angles)

    print("Convert the Euler angles to unit quaternions.")

    quaternions = np.apply_along_axis(pySymStat.conversion.euler_to_quaternion, axis = 1, arr = relion_euler_angles)
    print(quaternions)

    print("Convert the unit quaternions back to Euler angles.")

    relion_euler_angles = np.apply_along_axis(pySymStat.conversion.quaternion_to_euler, axis = 1, arr = quaternions)
    print(relion_euler_angles)

    print("===================================================")
    print("Generate a 2D array, each row is a unit quaternion.")

    quaternions = np.random.randn(N, 4)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    print(quaternions)

    print("Convert the unit quaternions to Euler angles.")

    relion_euler_angles = np.apply_along_axis(pySymStat.conversion.quaternion_to_euler, axis = 1, arr = quaternions)
    print(relion_euler_angles)

    print("Convert the Euler angles back to unit quaternions.")

    quaternions = np.apply_along_axis(pySymStat.conversion.euler_to_quaternion, axis = 1, arr = relion_euler_angles)
    print(quaternions)
