import pytest
import numba
import numpy as np
from hausdorff.hausdorff import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff

@pytest.fixture
def input_arrays():
    np.random.seed(42)
    XA = np.random.random((1000,5))
    XB = np.random.random((2000,5))
    return (XA,XB)

def test_it_computes_haussdorf_distance(input_arrays):
    XA,XB = input_arrays

    actual_dist = hausdorff_distance(XA, XB, distance="euclidean")

    assert isinstance(actual_dist, float)

def test_it_correctly_computes_haussdorf_distance(input_arrays):
    XA,XB = input_arrays

    actual_dist = hausdorff_distance(XA, XB, distance="euclidean")
    expected_dist = max(directed_hausdorff(XA, XB)[0], directed_hausdorff(XB, XA)[0])

    np.testing.assert_almost_equal(expected_dist, actual_dist)

def test_it_computes_custom_haussdorf_distance(input_arrays):
    XA,XB = input_arrays
    @numba.jit(nopython=True, fastmath=True)
    def custom_dist(array_x, array_y):
        n = array_x.shape[0]
        ret = 0.
        for i in range(n):
            ret += (array_x[i]-array_y[i])**3 / (array_x[i]**2 + array_y[i]**2 + 0.1)
        return ret
        
    actual_dist = hausdorff_distance(XA, XB, distance=custom_dist)

    assert isinstance(actual_dist, float)
