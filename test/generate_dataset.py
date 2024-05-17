import os
import numpy as np

if __name__ == '__main__':
    os.makedirs('data', exist_ok = True)

    # Generate enough many spatial rotations (represented by quaternions).
    rotation_dataset = np.random.randn(15000, 4)
    rotation_dataset /= np.linalg.norm(rotation_dataset, axis = 1)[:, None]
    np.save('data/SO3.npy', rotation_dataset)

    # Generate enough many projection directions (unit 3D-vectors).
    direction_dataset = np.random.randn(15000, 3)
    direction_dataset /= np.linalg.norm(direction_dataset, axis = 1)[:, None]
    np.save('data/S2.npy', direction_dataset)
