import numpy as np
from pySymStat.conversion import euler_to_quat, quat_to_euler, euler_to_vec, vec_to_euler

if __name__ == '__main__':
    N = 5

    print("======================================================")
    print("Generate a 2D array containing N spatial rotations in Euler angles form.")
    rot  = np.random.uniform(low = -180.0, high = 180.0, size = N)
    tilt = np.random.uniform(low =    0.0, high = 180.0, size = N)
    psi  = np.random.uniform(low = -180.0, high = 180.0, size = N)
    relion_euler_angles = np.stack((rot, tilt, psi))
    print(relion_euler_angles)

    print("Convert the Euler angles to unit quaternions.")
    quaternions = euler_to_quat(relion_euler_angles)
    print(quaternions)

    print("Convert the unit quaternions back to Euler angles.")
    relion_euler_angles = quat_to_euler(quaternions)
    print(relion_euler_angles)

    print("======================================================")
    print("Generate a 2D array containing N projection directions in Euler angles form.")
    rot  = np.random.uniform(low = -180.0, high = 180.0, size = N)
    tilt = np.random.uniform(low =    0.0, high = 180.0, size = N)
    relion_euler_angles = np.stack((rot, tilt))
    print(relion_euler_angles)

    print("Convert the Euler angles to unit vectors.")
    vectors = euler_to_vec(relion_euler_angles)
    print(vectors)

    print("Convert the unit quaternions back to Euler angles.")
    relion_euler_angles = vec_to_euler(vectors)
    print(relion_euler_angles)
