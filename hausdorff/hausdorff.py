import numpy as np
import numba
import hausdorff.distances as distances
from inspect import getmembers

def _find_available_functions(module_name):
	all_members = getmembers(module_name)
	available_functions = [member[0] for member in all_members 
						   if isinstance(member[1], numba.targets.registry.CPUDispatcher)]
	return available_functions

@numba.jit(nopython=True, fastmath=True)
def _hausdorff(XA, XB, distance_function):
	nA = XA.shape[0]
	nB = XB.shape[0]
	cmax = 0.
	for i in range(nA):
		cmin = np.inf
		for j in range(nB):
			d = distance_function(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = np.inf
		for i in range(nA):
			d = distance_function(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	return cmax

def hausdorff_distance(XA, XB, distance="euclidean"):
	assert distance in _find_available_functions(distances), 'Distance is not an implemented function'
	assert type(XA) is np.ndarray and type(XB) is np.ndarray, "Arrays must be of type numpy.ndarray"
	assert XA.ndim==2 and XB.ndim==2, "Arrays must be 2-dimensional"
	assert XA.shape[1]==XB.shape[0], "Arrays must have equal number of columns"
	if distance == 'haversine':
		assert XA.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
		assert XB.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
	distance_function = getattr(distances, distance)
	return _hausdorff(XA, XB, distance_function)
