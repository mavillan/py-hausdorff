#!python
#cython: cdivision=True 
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY as INF
from libc.math cimport abs, sqrt, pow, cos, sin, asin

ctypedef cnp.ndarray ndarray


#######################################################################################
# Defining distance functions
#######################################################################################

distances_mapping = {
	"manhattan": 1,
	"euclidean": 2,
	"chebyshev": 3,
	"cosine": 4,
	"haversine": 5
}


cdef inline double _manhattan(double[:] x, double[:] y):
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		double ret = 0.
	for i in range(n):
		ret += abs(x[i]-y[i])
	return ret


cdef inline double _euclidean(double[:] x, double[:] y):
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		double ret = 0.
	for i in range(n):
		ret += (x[i]-y[i])**2
	return sqrt(ret)


cdef inline double _chebyshev(double[:] x, double[:] y):
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		double ret = -1*INF
		double d 
	for i in range(n):
		d = abs(x[i]-y[i])
		if d>ret: ret=d
	return ret


cdef inline double _cosine(double[:] x, double[:] y):
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		double xydot = 0.
		double xnorm = 0.
		double ynorm = 0.
	for i in range(n):
		xydot += x[i]*y[i]
		xnorm += x[i]*x[i]
		ynorm += y[i]*y[i]
	return 1.-xydot/(sqrt(xnorm)*sqrt(ynorm))


cdef inline double _haversine(double [:] x, double [:] y):
	cdef:
		double R = 6378.0
		double radians = np.pi / 180.0
		double lat1 = radians * x[0]
		double lon1 = radians * x[1]
		double lat2 = radians * y[0]
		double lon2 = radians * y[1]
		double dlon = lon2 - lon1
		double dlat = lat2 - lat1
		double a

	a = (pow(sin(dlat/2.0), 2.0) + cos(lat1) *
		cos(lat2) * pow(sin(dlon/2.0), 2.0))

	return R * 2 * asin(sqrt(a))


#######################################################################################
# Defining Hausdorff functions
#######################################################################################


cdef inline double _hausdorff(double[:,:] XA, double[:,:] XB, int dist):
	cdef:
		Py_ssize_t nA = XA.shape[0]
		Py_ssize_t nB = XB.shape[0]
		Py_ssize_t i, j
		double cmax = 0.
		double cmin 
		double d

	for i in range(nA):
		cmin = INF
		for j in range(nB):
			if   dist==1: d = _manhattan(XA[i,:], XB[j,:])
			elif dist==2: d = _euclidean(XA[i,:], XB[j,:])
			elif dist==3: d = _chebyshev(XA[i,:], XB[j,:])
			elif dist==4: d = _cosine(XA[i,:], XB[j,:])
			elif dist==5: d = _haversine(XA[i,:], XB[j,:])

			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INF>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = INF
		for i in range(nA):
			if   dist==1: d = _manhattan(XA[i,:], XB[j,:])
			elif dist==2: d = _euclidean(XA[i,:], XB[j,:])
			elif dist==3: d = _chebyshev(XA[i,:], XB[j,:])
			elif dist==4: d = _cosine(XA[i,:], XB[j,:])
			elif dist==5: d = _haversine(XA[i,:], XB[j,:])

			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INF>cmin:
			cmax = cmin
	return cmax


def hausdorff(double[:,:] XA not None, double[:,:] XB not None, distance="euclidean"):
	assert distance in distances_mapping, "distance must be one of the following: " + " ".join(distances_mapping.keys())
	if distance == 'haversine':
		assert XA.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
		assert XB.shape[1] >= 2, 'Haversine distance requires at least 2 coordinates per point (lat, lng)'
	return _hausdorff(XA, XB, distances_mapping[distance])
