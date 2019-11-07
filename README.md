# py-hausdorff
[![PyPI version][pypi-image]][pypi-url]  [![PyPI download][download-image]][pypi-url]

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

Which computes the _Hausdorff distance_ between the rows of `X` and `Y` using the Euclidean distance as metric. It receives the optional argument `distance` (string), which is the distance function used to compute the distance between the rows of `X` and `Y`. It could be any of the following: `manhattan`, `euclidean` (default), `chebyshev` and `cosine`.

__Note:__ I will add more distances in the near future. If you need any distance in particular, open an issue. 

__Note:__ The haversine distance is calculated assuming lat, lng coordinate ordering and assumes
 the first two coordinates of each point are latitude and longitude respectively.

```python
import numpy as np
from hausdorff import hausdorff_distance

# two random 2D arrays (second dimension must match)
np.random.seed(0)
X = np.random.random((1000,100))
Y = np.random.random((5000,100))

# Test computation of Hausdorff distance with different base distances
print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="manhattan") ))
print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="euclidean") ))
print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="chebyshev") ))
print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="cosine") ))

# For haversine, use 2D lat, lng coordinates
def rand_lat_lng(N):
    lats = np.random.uniform(-90, 90, N)
    lngs = np.random.uniform(-180, 180, N)
    return np.stack([lats, lngs], axis=-1)
        
X = rand_lat_lng(100)
Y = rand_lat_lng(250)
print("Hausdorff haversine test: {0}".format( hausdorff_distance(X, Y, distance="haversine") ))
```

[pypi-image]: http://img.shields.io/pypi/v/hausdorff.svg
[pypi-url]: https://pypi.org/project/hausdorff/
[download-image]: http://img.shields.io/pypi/dm/hausdorff.svg

