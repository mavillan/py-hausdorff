#!python
#cython: cdivision=True 
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY
from libc.math cimport sqrt


ctypedef cnp.float64_t float64_t
ctypedef cnp.ndarray ndarray


cdef inline float64_t _metric(float64_t[::1] x, float64_t[::1] y) nogil:
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		float64_t ret = 0.
	for i in range(n):
		ret += (x[i]-y[i])**2
	return sqrt(ret)


cdef inline float64_t _weighted_metric(float64_t[::1] x, float64_t[::1] y, float64_t[::1] w) nogil:
	cdef:
		Py_ssize_t i
		Py_ssize_t n = x.shape[0]
		float64_t ret = 0.
	for i in range(n):
		ret += w[i]*(x[i]-y[i])**2
	return sqrt(ret)


cdef inline float64_t _hausdorff(float64_t[:,::1] XA, float64_t[:,::1] XB) nogil:
	cdef:
		Py_ssize_t nA = XA.shape[0]
		Py_ssize_t nB = XB.shape[0]
		Py_ssize_t i, j
		float64_t cmax = 0.
		float64_t cmin 
		float64_t d

	for i in range(nA):
		cmin = INFINITY
		for j in range(nB):
			d = _metric(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INFINITY>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = INFINITY
		for i in range(nA):
			d = _metric(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INFINITY>cmin:
			cmax = cmin
	return cmax


cdef inline float64_t _weighted_hausdorff(float64_t[:,::1] XA, float64_t[:,::1] XB, float64_t[::1] w) nogil:
	cdef:
		Py_ssize_t nA = XA.shape[0]
		Py_ssize_t nB = XB.shape[0]
		Py_ssize_t i, j
		float64_t cmax = 0.
		float64_t cmin 
		float64_t d

	for i in range(nA):
		cmin = INFINITY
		for j in range(nB):
			d = _weighted_metric(XA[i,:], XB[j,:], w)
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INFINITY>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = INFINITY
		for i in range(nA):
			d = _weighted_metric(XA[i,:], XB[j,:], w)
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and INFINITY>cmin:
			cmax = cmin
	return cmax


def hausdorff(float64_t[:,::1] XA not None, float64_t[:,::1] XB not None):
	return _hausdorff(XA, XB)


def weighted_hausdorff(XA not None, XB not None, w not None):
	return _weighted_hausdorff(XA, XB, w)