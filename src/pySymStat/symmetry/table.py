import numpy as np
from .generate import _quat_mult

def get_g_table(elems):
    products = _quat_mult(elems.T[:, :, np.newaxis], elems.T[:, np.newaxis, :])
    return np.argmax(np.square(np.tensordot(products, elems, axes = (0, -1))), axis = -1).astype(np.int32)

def get_i_table(g_table):
    return np.argmin(g_table, axis = -1).astype(np.int32)
