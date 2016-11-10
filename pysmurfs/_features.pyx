#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np


def spmeans(float[:, ::1] data, int[::1] labels):
	cdef int n
	cdef int N = labels.shape[0]
	cdef int K = data.shape[1]
	cdef int nsp = np.max(labels)+1
	cdef float[:, ::1] means = np.zeros((nsp, K), np.float32)
	cdef int[::1] sizes = np.zeros(nsp, np.int32)
	cdef int l, b
	cdef float t

	for n in range(N):
		l = labels[n]
		sizes[l] += 1

		for z in range(K):
			t = data[n, z]
			means[l, z] += t

	for n in range(nsp):
		if sizes[n] == 0:
			continue
		for z in range(K):
			means[n, z] /= sizes[n]

	return np.squeeze(np.asarray(means))
