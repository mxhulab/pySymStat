# **pySymStat 1.0**

**pySymStat** is a Python program package designed for solving statistic problem (mean and variance) of spatial rotations and projection directions when molecular symmetries are considered.

This program package implements algorithm proposed in the paper "*Statistics and classification of spatial rotations and projection directions considering molecular symmetry in 3D electron*", written by Mingxu HU, Qi ZHANG, Hai LIN. Users may consult our paper for more details.

We provide a demo program `demo/demo.py` which shows how to use our package for solving these two problems. As an example, we present how to solve mean and variance of spatial rotations.

### Step 1. Calculate Molecular Symmetry Group.

For solving mean and variance of some spatial rotations with molecular symmetries, we first need to obtain some knowledge about the molecular symmetry group.

```Python
    sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep = get_sym_grp('D7')
```

The function `get_sym_grp` receives a string representing a molecular symmetry group. We note that there are only five possible class of molecular symmetry groups, i.e., `CN`, `DN`, `T`, `O` and `I`, where `N` is a positive integer.  Then this function produces a 3-tuple containing the group information we need in the sequel, i.e., the group elements in unit quaternion form, the group multiplication table and the list of its irreducible representations.

### Step 2. Solving Mean and Variance.

After obtaining necessary information about the molecular symmetry group, we can call `spatial_rotation_mean_variance_SO3_G` for solving mean and variance of some given spatial rotations.

```Python
    mean, variance, representatives, sym_grp_rep = spatial_rotation_mean_variance_SO3_G(spatial_rotations, sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep)
```

The function `spatial_rotation_mean_variance_SO3_G` receives `spatial_rotations` and the tuple we get in first step as input. The `spatial_rotations` is a $n$ by $4$ numpy array, and $i$-th spatial rotation `spatial_rotations[i,:]` is in unit quaternion form. We note that this function returns not only mean and variance but also other informations like representatives. One may consult our paper for their meaning in details.

Similarly, one can call the function `projection_direction_mean_variance_S2_G` for solving mean and variance of projection directions with molecular symmetries.

```Python
    mean, variance, representatives, sym_grp_rep = projection_direction_mean_variance_S2_G(projection_directions, sym_grp_elems, sym_grp_table, sym_grp_irreducible_rep)
```

Here the parameter `projection_directions` is a $n$ by $3$ numpy array, and $i$-th projection direction $spatial_rotations[i,:]$ is a unit 3-vector.
