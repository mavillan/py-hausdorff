import numba
import numpy as np
from distances import *

@numba.jit(nopython=True, fastmath=True)
def _hausdorff(XA, XB, distance_func):
	nA = XA.shape[0]
	nB = XB.shape[0]
	cmax = 0.
	for i in range(nA):
		cmin = np.inf
		for j in range(nB):
			d = distance_func(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = np.inf
		for i in range(nA):
			d = distance_func(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	return cmax

def hausdorff(XA not None, XB not None, distance="euclidean"):
	assert distance in distances_mapping, "distance must be one of the following: " + " ".join(distances_mapping.keys())
    assert type(XA) is np.ndarray and type(XB) is np.ndarray, "Arrays must be of type numpy.ndarray"
    assert XA.ndim==2 and XB.ndim==2, "Arrays must be 2-dimensional"
    assert XA.shape[0]==XB.shape[0], "Arrays must have equal number of rows"
    assert XA.shape[1]==XB.shape[0], "Arrays must have equal number of columns"

	if distance == 'haversine':
		assert XA.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
		assert XB.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
	return _hausdorff(XA, XB, distances_mapping[distance])
