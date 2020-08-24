#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['projection_direction_mean_S2']

import numpy as np

def projection_direction_mean_S2(projection_directions):

    mean = np.mean(projection_directions, axis = 0)

    return mean / np.linalg.norm(mean)

if __name__ == '__main__':

    projection_directions = np.random.randn(100, 3)

    projection_directions = np.apply_along_axis(lambda x : x / np.linalg.norm(x), -1, projection_directions)

    print(projection_directions)
    print(np.linalg.norm(projection_directions, axis = -1))
    
    print(projection_direction_mean_S2(projection_directions))

    # print(projection_directions[0])
    # print(projection_directions[1])

    # print(tensor(projection_directions[0], projection_directions[1]))
