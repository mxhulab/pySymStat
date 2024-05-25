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

## `distance.distance_SO3`
```
distance_SO3(q1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], q2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> float
    The `distance_SO3` function calculates either the arithmetic or geometric distance between two spatial rotations.
    - `q1`, `q2`: These are the unit quaternion representations of spatial rotations, each a Numpy vector of type `np.float64` with a length of 4.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
```
[source](distance.py) [demo](demo/demo_distance_SO3.py)

## `distance.distance_S2`
```
distance_S2(v1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], v2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> float
    The `distance_S2` function calculates either the arithmetic or geometric distance between two projection directions.
    - `v1`, `v2`: These are the unit vectors representing projection directions, each a Numpy vector of type `np.float64` with a length of 3.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
```
[source](distance.py) [demo](demo/demo_distance_S2.py)

## `get_sym_grp`
```
get_sym_grp(sym)
    The `get_sym_grp` function retrieves the elements, the Cayley table, and the irreducible representations for a specified molecular symmetry symbol.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
```
[source](symmetry_group.py) [demo](demo/demo_get_sym_grp.py)

## `averaging_SO3.mean_SO3`
```
mean_SO3(quats: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic'] = 'arithmetic') -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
    The `mean_SO3` function calculates the mean of a set of spatial rotations.

    - `quats`: Unit quaternion representations of a set of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
```
[source](averaging_SO3.py) [demo](demo/demo_mean_SO3.py)

## `averaging_SO3.variance_SO3`
```
variance_SO3(quats: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic'] = 'arithmetic', mean: Literal[None, numpy.ndarray[Any, numpy.dtype[numpy.float64]]] = None) -> float
    The `variance_SO3` function calculates the variances of a set of spatial rotations.

    - `quats`: Unit quaternion representations of spatial rotations, provided as a numpy array with the shape `(n, 4)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input spatial rotations. If this is `None`, the variance is calculated in an unsupervised manner.
```
[source](averaging_SO3.py) [demo](demo/demo_variance_SO3.py)

## `averaging_S2.mean_S2`
```
mean_S2(vecs: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic'] = 'arithmetic') -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
    The `mean_S2` function calculates the mean of a set of projection directions.

    - `quats`: Unit vector representations of a set of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
```
[source](averaging_S2.py) [demo](demo/demo_mean_S2.py)

## `averaging_S2.variance_S2`
```
variance_S2(vecs: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], type: Literal['arithmetic'] = 'arithmetic', mean: Literal[None, numpy.ndarray[Any, numpy.dtype[numpy.float64]]] = None) -> float
    The `variance_S2` function calculates the variances of a set of projection directions.

    - `vecs`: Unit vector representations of projection directions, provided as a numpy array with the shape `(n, 3)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input projection directions. If this is `None`, the variance is calculated in an unsupervised manner.
```
[source](averaging_S2.py) [demo](demo/demo_variance_S2.py)
