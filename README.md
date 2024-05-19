# **pySymStat Overivew**

pySymStat is a Python software package designed to average orientations, including both spatial rotations and projection directions, while accounting for molecular symmetry.

# Publications

This program package implements algorithm proposed in the paper [The Moments of Orientation Estimations Considering Molecular Symmetry in Cryo-EM](https://arxiv.org/abs/2301.05426).

# Installation

CryoSieve is an open-source software, developed using Python. Please access our source code on [GitHub](https://github.com/mxhulab/pySymStat).

## Preparation of Conda Environment

Prepare a conda environment with the following commands
```
conda create -n PYSYMSTAT_ENV python=3.10
```
and activate this conda environemnt via
```
conda activate PYSYMSTAT_ENV
```

## Installing Dependencies

Install Numpy and PICOS by executing the following command with Pip:
```
pip install numpy picos
```

## Installing pySymStat

Clone the Github repository of [pySymStat](https://github.com/mxhulab/pySymStat).

## Verifying Installation

Open your Python environment, and execute the following command to import the `pySymStat` package:
```
import pySymStat
```
Then, list all functions, classes, and modules included in `pySymStat` by running:
```
print(dir(pySymStat))
```

# Tutorial

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

# Function List

# Function List

## `distance.distance_SO3`
```
distance_SO3(q1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], q2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type=typing.Literal['arithmetic', 'geometric']) -> float
    q1, q2 : unit quaternion representation of spatial rotations, Numpy vector of np.float64 of length 4.
    type : 'arithmetic' | 'geometric'.
```

## `distance.distance_S2`
