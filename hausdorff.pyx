#!python
#cython: cdivision=True 
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY as INF
from libc.math cimport abs,sqrt

ctypedef cnp.ndarray ndarray


#######################################################################################
# Defining distance functions
#######################################################################################

distances_mapping = {"manhattan":1, "euclidean":2, "chebyshev":3, "cosine":4}


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

			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INF>cmin:
			cmax = cmin
	return cmax


def hausdorff(double[:,:] XA not None, double[:,:] XB not None, distance="euclidean"):
	assert distance in distances_mapping, "distance must be one of the following: " + " ".join(distances_mapping.keys())	
	return _hausdorff(XA, XB, distances_mapping[distance])
