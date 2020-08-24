__all__ = ['quat_norm', \
           'quat_conj', \
           'quat_mult', \
           'quat_same', \
           'rotate']

import numpy as np

EQUAL_ACCURACY = 1e-2

def quat_norm(quat):

    return np.linalg.norm(quat)

def quat_conj(quat):

    cq = quat.copy()

    cq[1:] *= -1

    return cq

def quat_mult(quat1, quat0):

    w0, x0, y0, z0 = quat0
    w1, x1, y1, z1 = quat1

    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, \
                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, \
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, \
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype = np.float64)

def quat_same(quat0, quat1):

    w0, x0, y0, z0 = quat0
    w1, x1, y1, z1 = quat1

    if ((np.abs(w0 - w1) < EQUAL_ACCURACY) and \
        (np.abs(x0 - x1) < EQUAL_ACCURACY) and \
        (np.abs(y0 - y1) < EQUAL_ACCURACY) and \
        (np.abs(z0 - z1) < EQUAL_ACCURACY)):
        return True

    if ((np.abs(w0 + w1) < EQUAL_ACCURACY) and \
        (np.abs(x0 + x1) < EQUAL_ACCURACY) and \
        (np.abs(y0 + y1) < EQUAL_ACCURACY) and \
        (np.abs(z0 + z1) < EQUAL_ACCURACY)):
        return True

    return False

# vq, vector in quaternion form
def rotate(quat, v):

    vq = np.array([0, v[0], v[1], v[2]])

    return quat_mult(quat_mult(quat, vq), quat_conj(quat))[1:]

if __name__ == '__main__':

    print(quat_same(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])))

    print(quat_same(np.array([0, 0, 0, 1]), np.array([0, 0, 0, -1])))

    print(quat_same(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 2])))
