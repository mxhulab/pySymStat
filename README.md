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

The function [`averaging_SO3_G`](#averaging_so3_gmean_variance_so3_g) calculates the mean and variance of a given set of spatial rotations represented as unit quaternions. Meanwhile, the function [`averaging_S2_G`](#averaging_s2_gmean_variance_s2_g) computes the mean and variance of a given set of projection directions represented as unit vectors.

Click the hyperlinks to view the descriptions, help information, source code, and demo script for these two functions.

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

## `averaging_SO3_G.mean_variance_SO3_G`
```
mean_variance_SO3_G(quats: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], sym_grp, type: Literal['arithmetic'] = 'arithmetic', **kwargs)
    The `mean_variance_SO3_G` function calculates the mean and variance of a set of spatial rotations with molecular symmetry.

    - `quats`: Unit quaternion representations of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.

    Output:
    - `output[0]`: The mean of these spatial rotations.
    - `output[1]`: The variance of these spatial rotations.
    - `output[2]`: The correct representatives of these spatial rotations.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
```
[source](averaging_SO3_G.py) [demo](demo/demo_mean_variance_SO3_G.py)

## `averaging_S2_G.mean_variance_S2_G`
```
mean_variance_S2_G(vecs: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], sym_grp, type: Literal['arithmetic'] = 'arithmetic', **kwargs)
    The `mean_variance_S2_G` function calculates the mean and variance of a set of projection directions with molecular symmetry.

    - `quats`: Unit quaternion representations of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in Relion. For more details, visit [Relion Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.

    Output:
    - `output[0]`: The mean of these projection directions.
    - `output[1]`: The variance of these projection directions.
    - `output[2]`: The correct representatives of these projection directions.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
```
[source](averaging_S2_G.py) [demo](demo/demo_mean_variance_S2_G.py)
