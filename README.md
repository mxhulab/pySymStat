# pySymStat Overview

pySymStat is a Python software package designed to average orientations, including both spatial rotations and projection directions, while accounting for molecular symmetry.

# Publications

This program package implements algorithm proposed in the paper [Averaging Orientations with Molecular Symmetry in Cryo-EM](https://arxiv.org/abs/2301.05426).

# Installation

pySymStat is an open-source software, developed using Python. Please access our source code on [GitHub](https://github.com/mxhulab/pySymStat).

## Installing pySymStat

Install pySymStat by executing the following command with Pip:
```
pip install pysymstat
```

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

The function [`mean_variance_SO3_G`](#averaging_so3_gmean_variance_so3_g) calculates the mean and variance of a given set of spatial rotations represented as unit quaternions with molecular symmetry. Meanwhile, the function [`mean_variance_S2_G`](#averaging_s2_gmean_variance_s2_g) computes the mean and variance of a given set of projection directions represented as unit vectors with molecular symmetry. Click the hyperlinks to view the descriptions, help information, source code, and demo script for these two functions.

A set of functions related to these calculations is also provided in this package. Please refer to the Function List section for a comprehensive list.

# Function List

## `conversion.euler_to_quat`
```
euler_to_quat(src: numpy.ndarray) -> numpy.ndarray
    Converts spatial rotations in Euler angles form (RELION's convention) to unit quaternion form.
    - `src`: This is a NumPy vector of type `np.float64` with a length of 3, representing `rlnAngleRot`, `rlnAngleTilt`, and `rlnAnglePsi` in RELION's starfile convention.
```
[source](src/pySymStat/conversion.py) [demo](demo/demo_conversion.py)

## `conversion.quat_to_euler`
```
quat_to_euler(src: numpy.ndarray) -> numpy.ndarray
    Converts spatial rotations in unit quaternion form to Euler angles form (RELION's convention).
    - `src`: This is a NumPy vector of type `np.float64` with a length of 4, which is a quaternion representing a spatial rotation.
```
[source](src/pySymStat/conversion.py) [demo](demo/demo_conversion.py)

## `conversion.euler_to_vec`
```
euler_to_vec(src: numpy.ndarray) -> numpy.ndarray
    Converts projection directions in Euler angles form (RELION's convention) to unit vector form.
    - `src`: This is a NumPy vector of type `np.float64` with a length of 3, representing `rlnAngleRot`, `rlnAngleTilt`, and `rlnAnglePsi` in RELION's starfile convention.
```
[source](src/pySymStat/conversion.py) [demo](demo/demo_conversion.py)

## `conversion.vec_to_euler`
```
vec_to_euler(src: numpy.ndarray) -> numpy.ndarray
    Converts projection directions in unit vector form to Euler angles form (RELION's convention).
    - `src`: This is a NumPy vector of type `np.float64` with a length of 4, which is a quaternion representing a spatial rotation.
```
[source](src/pySymStat/conversion.py) [demo](demo/demo_conversion.py)

## `quaternion.quat_conj`
```
quat_conj(q: numpy.ndarray) -> numpy.ndarray
    Calculates the conjugate of an input quaternion.
    - `q`: This is a quaternion and should be a NumPy vector of type `np.float64` with a length of 4.
```
[source](src/pySymStat/quaternion.py) [demo](demo/demo_quat_conj.py)

## `quaternion.quat_mult`
```
quat_mult(q1: numpy.ndarray, q2: numpy.ndarray) -> numpy.ndarray
    Calculates the product of two input quaternions.
    - `q1` and `q2`: This are two quaternions and should be NumPy vectors of type `np.float64` with a length of 4.
```
[source](src/pySymStat/quaternion.py) [demo](demo/demo_quat_mult.py)

## `quaternion.quat_rotate`
```
quat_rotate(q: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray
    Rotate a vector based on a spatial rotation represented by a unit quarternion.
    - `q`: This is a quaternion and should be a NumPy vector of type `np.float64` with a length of 4. It represents the spatial rotation.
    - `v`: This is a vector in 3D space, which to be rotated.
```
[source](src/pySymStat/quaternion.py) [demo](demo/demo_quat_rotate.py)

## `distance_SO3`
```
distance_SO3(q1: numpy.ndarray, q2: numpy.ndarray, type: Literal['arithmetic', 'geometric'] = 'arithmetic', pairwise : bool = False) -> float
    Calculates either the arithmetic or geometric distance between two (set of) spatial rotations.
    - `q1`, `q2`: These are the unit quaternion representations of spatial rotations, each a NumPy vector of type `np.float64` with a length of 4.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
    - `pairwise`: Specifies whether in broadcast mode or pairwise mode.
```
[source](src/pySymStat/distance.py) [demo](demo/demo_distance_SO3.py)

## `distance_S2`
```
distance_S2(v1: numpy.ndarray, v2: numpy.ndarray, type: Literal['arithmetic', 'geometric'] = 'arithmetic', pairwise : bool = False) -> float
    Calculates either the arithmetic or geometric distance between two projection directions.
    - `v1`, `v2`: These are the unit vectors representing projection directions, each a NumPy vector of type `np.float64` with a length of 3.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.
    - `pairwise`: Specifies whether in broadcast mode or pairwise mode.
```
[source](src/pySymStat/distance.py) [demo](demo/demo_distance_S2.py)

## `symmetry.Symmetry`
```
Symmetry(sym)
    The `Symmetry` class retrieves all information of a molecular symmetry group. It contains the group elements (in unit quaternion descriptions), the Cayley table, the inverse table, and the real irreducible representations.
    - `sym`: The molecular symmetry symbol. Acceptable inputs include `Cn`, `Dn`, `T`, `O`, `I`, `I1`, `I2`, `I3`. The symbols `I`, `I1`, `I2`, `I3` all denote icosahedral symmetry, but with different conventions. Notably, `I` is equivalent to `I2`. This convention is used in RELION. For more details, visit [RELION Conventions](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#symmetry).
```
[source](src/pySymStat/symmetry/) [demo](demo/demo_symmetry.py)

## `distance_SO3_G`
```
distance_SO3_G(q1: numpy.ndarray, q2: numpy.ndarray, sym : Union[str, Symmetry], type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Calculates either the arithmetic or geometric distance between two spatial rotations with molecular symmetry.
    - `q1`, `q2`: These are the unit quaternion representations of spatial rotations, each a NumPy vector of type `np.float64` with a length of 4.
    - `sym`: The molecular symmetry group.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.

    Output:
    - `output[0]`: The distance between two spatial rotations with molecular symmetry.
    - `output[1]`: The index of group element action on `q2` to make it closest to `q1`.
    - `output[2]`: The group element action on `q2` to make it closest to `q1`.
    - `output[3]`: The closest representative of `q2` to `q1`.
```
[source](src/pySymStat/distance.py) [demo](demo/demo_distance_SO3_G.py)

## `distance_S2_G`
```
distance_S2_G(v1: numpy.ndarray, v2: numpy.ndarray, sym : Union[str, Symmetry], type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Calculates either the arithmetic or geometric distance between two projection directions with molecular symmetry.
    - `v1`, `v2`: These are the unit vectors representing projection directions, each a NumPy vector of type `np.float64` with a length of 3.
    - `sym`: The molecular symmetry group.
    - `type`: Specifies the type of distance calculation. Options are 'arithmetic' or 'geometric'.

    Output:
    - `output[0]`: The distance between two projection direction with molecular symmetry.
    - `output[1]`: The index of group element action on `v2` to make it closest to `v1`.
    - `output[2]`: The group element action on `v2` to make it closest to `v1`.
    - `output[3]`: The closest representative of `v2` to `v1`
```
[source](src/pySymStat/distance.py) [demo](demo/demo_distance_S2_G.py)

## `mean_SO3`
```
mean_SO3(quats: numpy.ndarray, type: Literal['arithmetic'] = 'arithmetic') -> numpy.ndarray
    Calculates the mean of a set of spatial rotations.
    - `quats`: Unit quaternion representations of a set of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_mean_SO3.py)

## `variance_SO3`
```
variance_SO3(quats: numpy.ndarray, type: Literal['arithmetic'] = 'arithmetic', mean: Optional[numpy.ndarray] = None) -> float
    Calculates the variances of a set of spatial rotations.
    - `quats`: Unit quaternion representations of spatial rotations, provided as a numpy array with the shape `(n, 4)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input spatial rotations. If this is `None`, the variance is calculated in an unsupervised manner.
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_variance_SO3.py)

## `mean_S2`
```
mean_S2(vecs: numpy.ndarray, type: Literal['arithmetic'] = 'arithmetic') -> numpy.ndarray
    Calculates the mean of a set of projection directions.
    - `quats`: Unit vector representations of a set of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `type`: Specifies the type of distance. Only accepts the value `arithmetic`."
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_mean_S2.py)

## `variance_S2`
```
variance_S2(vecs: numpy.ndarray, type: Literal['arithmetic'] = 'arithmetic', mean: Optional[numpy.ndarray] = None) -> float
    Calculates the variances of a set of projection directions.
    - `vecs`: Unit vector representations of projection directions, provided as a numpy array with the shape `(n, 3)` and a data type of `np.float64`.
    - `type`: Specifies the type of distance calculation to be used. It only accepts the value `arithmetic`.
    - `mean`: Specifies the mean of the input projection directions. If this is `None`, the variance is calculated in an unsupervised manner.
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_variance_S2.py)

## `mean_variance_SO3_G`
```
mean_variance_SO3_G(quats: numpy.ndarray, sym : Union[str, Symmetry], type: Literal['arithmetic', 'geometric'] = 'arithmetic', **kwargs) -> Tuple[Optional[numpy.ndarray], Optional[float], float, numpy.ndarray, numpy.ndarray]
    Calculates the mean and variance of a set of spatial rotations with molecular symmetry.
    - `quats`: Unit quaternion representations of spatial rotations. It is a numpy array of shape `(n, 4)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry group.
    - `type`: Specifies the type of distance calculation to be used.

    Output:
    - `output[0]`: The mean of these spatial rotations.
    - `output[1]`: The (supervised version) variance of these spatial rotations.
    - `output[2]`: The optimal cost (unsupervised version variance) of NUG problem.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
    - `output[4]`: The representatives.
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_mean_variance_SO3_G.py)

## `mean_variance_S2_G`
```
mean_variance_S2_G(vecs: numpy.ndarray, sym : Union[str, Symmetry], type: Literal['arithmetic', 'geometric'] = 'arithmetic', **kwargs)
    Calculates the mean and variance of a set of projection directions with molecular symmetry.
    - `quats`: Unit quaternion representations of projection directions. It is a numpy array of shape `(n, 3)` with a data type of `np.float64`.
    - `sym`: The molecular symmetry group.
    - `type`: Specifies the type of distance calculation to be used.

    Output:
    - `output[0]`: The mean of these projection directions.
    - `output[1]`: The (supervised version) variance of these projection directions.
    - `output[2]`: The optimal cost (unsupervised version variance) of NUG problem.
    - `output[3]`: The index of elements in the symmetry group corresponding to the correct representative.
    - `output[4]`: The representatives.
```
[source](src/pySymStat/meanvar.py) [demo](demo/demo_mean_variance_S2_G.py)

# Test Data and Reproduction of Results

In the `test` folder of the `pySymStat` repository, we provide the dataset and expected results for our simulation experiments. Interested researchers can utilize this data to reproduce the results presented in Section 4.1 of our paper.

Please note that the contents of the `test` folder are not included in the PyPI release version of `pySymStat`. Users are required to clone the repository into their local directory using the following command:
```
git clone https://github.com/mxhulab/pySymStat.git
```
Alternatively, users can download the Release source code file and extract it.

To rerun the test, navigate to the `test` directory and extract the `data.zip` file into a folder named `data`. Then, execute the `run.sh` script with the following command:
```
cd test
unzip data.zip -d data
bash run.sh
```
Please note that the entire testing process may take a considerable amount of time to complete. We have provided all intermediate results in the `data.zip` file. If you only wish to view the summary result, you can run the `summary.py` script using the following command:
```
python summary.py
```
This will print a summary of all the tests as follows:
```
Approximation ability of \tilde{L}^{SO(3)} to L^{SO(3)}.
  For group C2, RoE = 33.7%, Pr[RCG < 0.01] = 50.6%, Pr[RCG < 0.1] = 95.3%.
  For group C7, RoE = 44.8%, Pr[RCG < 0.01] = 60.5%, Pr[RCG < 0.1] = 94.2%.
  For group D2, RoE = 93.4%, Pr[RCG < 0.01] = 96.7%, Pr[RCG < 0.1] = 100.0%.
  For group D7, RoE = 96.1%, Pr[RCG < 0.01] = 99.1%, Pr[RCG < 0.1] = 100.0%.
  For group T , RoE = 98.4%, Pr[RCG < 0.01] = 99.9%, Pr[RCG < 0.1] = 100.0%.
  For group O , RoE = 98.6%, Pr[RCG < 0.01] = 100.0%, Pr[RCG < 0.1] = 100.0%.
  For group I , RoE = 99.9%, Pr[RCG < 0.01] = 100.0%, Pr[RCG < 0.1] = 100.0%.

Approximation ability of \tilde{L}^{S2} to L^{S2}.
According to THEOREM 2.2, RoE should be 100%.
  For group C2, RoE = 100.0%.
  For group C7, RoE = 100.0%.
  For group D2, RoE = 100.0%.
  For group D7, RoE = 100.0%.
  For group T , RoE = 100.0%.
  For group O , RoE = 100.0%.
  For group I , RoE = 100.0%.

Plot scatter point graph of (\tilde{L}^{S^2}, L^{S^2}).
Plot scatter point graph of (\tilde{L}^{SO(3)}, L^{SO(3)}).

Approximation ability of NUG approach with our rounding for \tilde{L}^{SO(3)}.
Show (RoE, max-RCG).
| Group |   d_{SO(3)}^A   |   d_{SO(3)}^G   |    d_{S2}^A     |    d_{S2}^G     |
|-------------------------------------------------------------------------------|
|   C2  | ( 98.1%, 0.82%) | ( 98.7%, 1.32%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|   C7  | ( 99.9%, 0.22%) | ( 98.7%, 2.33%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|   D2  | ( 99.9%, 0.13%) | (100.0%, 0.00%) | ( 99.9%, 0.02%) | (100.0%, 0.00%) |
|   D7  | (100.0%, 0.00%) | ( 99.9%, 0.58%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|   T   | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|   O   | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|   I   | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) | (100.0%, 0.00%) |
|-------------------------------------------------------------------------------|

The (RoE, max-RCG) results for varying $m$ under $d_{SO(3)}^A$.
|-------------------------------------------------------------------------------|
| Group |   m=12, c=0.99  |   m=4, c=0.99   |   m=20, c=0.5   |    m=20, c=0    |
|   C2  | ( 98.1%, 0.82%) | ( 98.0%, 0.82%) | ( 89.6%, 2.40%) | ( 89.6%, 2.40%) |
|   C7  | ( 99.8%, 0.50%) | ( 97.9%, 8.27%) | ( 95.7%, 2.48%) | ( 95.3%, 2.48%) |
|   D2  | ( 99.9%, 0.13%) | ( 99.9%, 0.13%) | ( 94.6%, 8.74%) | ( 94.6%, 8.74%) |
|   D7  | (100.0%, 0.00%) | ( 99.8%, 1.62%) | ( 96.8%, 13.11%) | ( 96.5%, 17.62%) |
|   T   | (100.0%, 0.00%) | (100.0%, 0.00%) | ( 98.2%, 33.32%) | ( 98.2%, 33.32%) |
|   O   | (100.0%, 0.00%) | (100.0%, 0.00%) | ( 99.2%, 6.89%) | ( 99.2%, 6.89%) |
|   I   | (100.0%, 0.00%) | (100.0%, 0.00%) | ( 99.7%, 2.00%) | ( 99.7%, 2.00%) |
|-------------------------------------------------------------------------------|
```
