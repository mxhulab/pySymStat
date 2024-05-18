# **pySymStat Overivew**

pySymStat is a Python program package designed for averaging orientations (both spatial rotations and projection directions) with molecular symmetry.

# Publications

This program package implements algorithm proposed in the paper [The Moments of Orientation Estimations Considering Molecular Symmetry in Cryo-EM](https://arxiv.org/abs/2301.05426).

# Installation

CryoSieve is an open-source software, developed using Python. Please access our source code on [GitHub](https://github.com/mxhulab/pySymStat).

## Preparation of Conda environment



We provide a demo program `demo/demo.py` which shows how to use our package for solving these two problems. As an example, we present how to solve mean and variance of spatial rotations.

```Python
    mean, variance, representatives, solutions = meanvar_SO3_G(quats, sym_grp, type = 'arithmetic')
```

Here `quats` is a $n$ by $4$ numpy array, and `quats[i, :]` is the unit quaternion representation of $i$-th spatial rotation. `sym_grp` is a string representing a molecular symmetry group. Note that there are only five possible class of molecular symmetry groups, i.e., `CN`, `DN`, `T`, `O` and `I`, where `N` is a positive integer. There are some other options, e.g., the type of distance can be `arithmetic` or `geometric`. Then this function returns not only mean and variance but also other informations like representatives and optimal group elements. One may consult our paper for their meaning in details.

Similarly, one can call the function `projection_direction_mean_variance_S2_G` for solving mean and variance of projection directions with molecular symmetries.
```Python
    mean, variance, representatives, solutions = meanvar_S2_G(vecs, sym_grp, type = 'arithmetic')
```
Here `vecs` is a $n$ by $3$ numpy array, and $i$-th projection direction $vecs[i, :]$ is a unit 3-vector.
