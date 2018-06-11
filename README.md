# py-hausdorff
Fast computation of Hausdorff distance in Python/Cython. 

This code implements the algorithm presented in _An Efficient Algorithm for Calculating the Exact Hausdorff Distance_ (__DOI:__ [10.1109/TPAMI.2015.2408351](https://doi.org/10.1109/TPAMI.2015.2408351)) by _Aziz and Hanbury_.


To install the package I provide you a `setup.py` file. You must run:

```bash
python setup.py install
```

The main functions is: 

`hausdorff(np.ndarray[:,:] X, np.ndarray[:,:] Y)`

Which computes the _Hausdorff distance_ between the rows of `X` and `Y` using the Euclidean distance as metric. It receives the optional argument `distance` (string), which is the distance function used to compute the distance between the rows of `X` and `Y`. It could be any of the following: `manhattan`, `euclidean` (default), `chebyshev` and `cosine`.

__Note:__ I will add more distances in the near future. If you need any distance in particular raise an issue. 

```python
import numpy as np
from hausdorff import hausdorff

# two random 2D arrays (second dimension must match)
np.random.seed(0)
X = np.random.random((1000,100))
Y = np.random.random((5000,100))

# Test computation of Hausdorff distance with different base distances
print("Hausdorff distance test: {0}".format( hausdorff(X, Y, distance="manhattan") ))
print("Hausdorff distance test: {0}".format( hausdorff(X, Y, distance="euclidean") ))
print("Hausdorff distance test: {0}".format( hausdorff(X, Y, distance="chebyshev") ))
print("Hausdorff distance test: {0}".format( hausdorff(X, Y, distance="cosine") ))

```
