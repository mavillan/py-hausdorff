# py-hausdorff
Fast computation of hausdorff distance in Python/Cython. 

This code implements the algorithm presented in [An Efficient Algorithm for Calculating the Exact Hausdorff Distance](https://publik.tuwien.ac.at/files/PubDat_247739.pdf) (_Aziz and Hanbury_).

There are two main functions: 

1. `hausdorf(np.ndarray[:,:] X, np.ndarray[:,:] Y)` and
2. `weighted_hausdorff(np.ndarray[:,:] X, np.ndarray[:,:] Y, np.ndarray[:] w)`. 

The first computes the _Hausdorff distance_ between the rows of `X` and `Y` using the Euclidean distance as metric. The second computes the _Hausdorff distance_ between the rows of `X` and `Y` using a Weighted Euclidean distance as metric (with weights `w`). 

```python
import numpy as np
from hausdorff import hausdorff, weighted_hausdorff

# two random 2D arrays (second dimension must match)
X = np.random.random((1000,100))
Y = np.random.random((5000,100))

# 1D weights array
w = np.random.random(100)

print("Hausdorff distance test: {0}".format(hausdorff(X, Y)))
print("Weighted Hausdorff distance test: {0}".format(weighted_hausdorff(X, Y, w)))
```
