import pytest
import numpy as np
from hausdorff.hausdorff import hausdorff_distance

@pytest.fixture
def input_arrays():
    np.random.seed(42)
    XA = np.random.random((1000,5))
    XB = np.random.random((2000,5))
    return (XA,XB)

@pytest.mark.parametrize("distance", [
    "manhattan",
    "euclidean",
    "chebyshev",
    "cosine",
    "haversine",
])
def test_it_computes_haussdorf_distance(input_arrays, distance):
    XA,XB = input_arrays

    actual_dist = hausdorff_distance(XA, XB, distance=distance)

    assert isinstance(actual_dist, float)