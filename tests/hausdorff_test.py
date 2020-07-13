import pytest
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
