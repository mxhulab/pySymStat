#!/usr/bin/env python

import os, sys
import random
import math
import numpy as np
import time
import sys

__all__ = ['projection_direction_cluster']

from .distance_S2_G import *

from .mean_variance_sym_grp import *

def projection_direction_cluster(projection_directions, \
                                 num_classes, \
                                 sym_grp_elems, \
                                 sym_grp_table, \
                                 sym_grp_irreducible_rep, \
                                 jump_ratio_threshold = 0.2, \
                                 silence = False):

    class_ids = np.random.randint(0, num_classes, projection_directions.shape[0], dtype = np.uint32)

    means = np.random.randn(num_classes, 3)
    means = np.apply_along_axis(lambda v : v / np.linalg.norm(v), -1, means)

    reps = np.empty((projection_directions.shape[0], 3), dtype = np.float64)
    grps = np.empty((projection_directions.shape[0], 4), dtype = np.float64)

    i_round = 0

    while True:

        pre_class_ids = class_ids.copy()

        distances = np.empty((projection_directions.shape[0], means.shape[0]), dtype = np.float64)

        time_0 = time.time()

        for i, j in np.ndindex(distances.shape):
            distances[i, j] = distance_S2_G(projection_directions[i], means[j], sym_grp_elems)[0]

        time_1 = time.time()

        if not silence:
            print("TIME CONSUMPTION, ROUND {}, GET DISTANCE MATRICES: {:.2f}s".format(i_round, time_1 - time_0))

        for k in range(num_classes):

            num_elems = projection_directions.shape[0] // num_classes
            if k < projection_directions.shape[0] % num_classes:
                num_elems += 1

            indices = np.argsort(distances[:, k])[:num_elems]

            class_ids[indices] = k
            distances[indices, :] = np.nan
            # print(distances)

        # class_ids = np.argmin(distances, axis = -1)

        inequivalence_counter = len(np.nonzero(class_ids - pre_class_ids)[0])

        if not silence:
            print("CLUSTERING JUMP RATIO, ROUND {}, {:.2f}%".format(i_round, 100 * float(inequivalence_counter) / projection_directions.shape[0]))

        if (inequivalence_counter < projection_directions.shape[0] * jump_ratio_threshold):
            break

        time_0 = time.time()

        for k in range(num_classes):

            class_projection_directions = projection_directions[class_ids == k]


            means[k], _, reps[class_ids == k], grps[class_ids == k] = projection_direction_mean_variance_S2_G(class_projection_directions, sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep)

            time_1 = time.time()

        if not silence:
            print("TIME CONSUMPTION, ROUND {}, DETERMINING MEAN OF EACH CLUSTER: {:.2f}s".format(i_round, time_1 - time_0))

        i_round += 1

    return means, class_ids, reps, grps
