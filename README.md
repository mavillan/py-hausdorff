# py-hausdorff
[![Build Status][travis-image]][travis-url]  [![PyPI version][pypi-image]][pypi-url]  [![PyPI download][download-image]][pypi-url]

Fast computation of Hausdorff distance in Python. 

This code implements the algorithm presented in _An Efficient Algorithm for Calculating the Exact Hausdorff Distance_ (__DOI:__ [10.1109/TPAMI.2015.2408351](https://doi.org/10.1109/TPAMI.2015.2408351)) by _Aziz and Hanbury_.


## Installation

Via [PyPI](https://pypi.org/project/hausdorff/):

```bash
pip install hausdorff
```
Or you can clone this repository and install it manually: 

```bash
python setup.py install
```

## Example Usage
The main functions is: 

`hausdorff_distance(np.ndarray[:,:] X, np.ndarray[:,:] Y)`

Which computes the _Hausdorff distance_ between the rows of `X` and `Y` using the Euclidean distance as metric. It receives the optional argument `distance` (**string or callable**), which is the distance function used to compute the distance between the rows of `X` and `Y`. In case of string, it could be any of the following: `manhattan`, `euclidean` (default), `chebyshev` and `cosine`. In case of callable, it should be a numba decorated function (see example below).


__Note:__ The haversine distance is calculated assuming lat, lng coordinate ordering and assumes
 the first two coordinates of each point are latitude and longitude respectively.
 
 ### Basic Usage

```python
import numpy as np
from hausdorff import hausdorff_distance

# two random 2D arrays (second dimension must match)
np.random.seed(0)
X = np.random.random((1000,100))
Y = np.random.random((5000,100))

# Test computation of Hausdorff distance with different base distances
print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='manhattan')}")
print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='euclidean')}")
print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='chebyshev')}")
print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='cosine')}")

# For haversine, use 2D lat, lng coordinates
def rand_lat_lng(N):
    lats = np.random.uniform(-90, 90, N)
    lngs = np.random.uniform(-180, 180, N)
    return np.stack([lats, lngs], axis=-1)
        
X = rand_lat_lng(100)
Y = rand_lat_lng(250)
print("Hausdorff haversine test: {0}".format( hausdorff_distance(X, Y, distance="haversine") ))
```

### Custom distance function

The distance function is used to calculate the distances between the rows of the input 2-dimensional arrays . For optimal performance, this custom distance function should be decorated with `@numba` in [nopython mode](https://numba.pydata.org/numba-doc/latest/user/jit.html).

```python
import numpy as np
from math import sqrt
from hausdorff import hausdorff_distance

# two random 2D arrays (second dimension must match)
np.random.seed(0)
X = np.random.random((1000,100))
Y = np.random.random((5000,100))

# write your own crazy custom function here
# this function should take two 1-dimensional arrays as input
# and return a single float value as output.
@numba.jit(nopython=True, fastmath=True)
def custom_func(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i]-array_y[i])**2
    return sqrt(ret)

print(f"Hausdorff custom euclidean test: {hausdorff_distance(XA, XB, distance=custom_dist)}")

# a real crazy custom function
@numba.jit(nopython=True, fastmath=True)
def custom_dist(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i]-array_y[i])**3 / (array_x[i]**2 + array_y[i]**2 + 0.1)
    return ret

print(f"Hausdorff custom crazy test: {hausdorff_distance(XA, XB, distance=custom_dist)}")
```

[travis-image]: https://travis-ci.org/mavillan/py-hausdorff.svg?branch=master
[travis-url]: https://travis-ci.org/mavillan/py-hausdorff
[pypi-image]: http://img.shields.io/pypi/v/hausdorff.svg
[pypi-url]: https://pypi.org/project/hausdorff/
[download-image]: http://img.shields.io/pypi/dm/hausdorff.svg
