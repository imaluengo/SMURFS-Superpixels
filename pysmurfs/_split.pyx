#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

from cpython cimport bool
from libc.math cimport sqrt, floor
from libcpp.vector cimport vector


def extract_regions(int[::1] regions):
	cdef int N = regions.shape[0]
	cdef int i, r

	cdef dict reginfo = {}

	for i in range(N):
		r = regions[i]
		if r in reginfo:
			reginfo[r].append(i)
		else:
			reginfo[r] = [i]

	return reginfo


cpdef bool initial_centroids(float[:, ::1] data, float[:, ::1] centers, int K,
							 int nsplit=50):
	cdef int N = data.shape[0]
	cdef int C = data.shape[1]

	cdef int[::1] clusters = np.zeros(N, np.int32)
	cdef int[::1] sizes = np.zeros(K, np.int32)

	cdef int n, c, k

	cdef bool split_cluster = True
	for k in range(K-1):
		split_cluster = find_split(data, clusters, k+1, nsplit)
		if not split_cluster:
			return False

	for n in range(N):
		k = clusters[n]
		sizes[k] += 1
		for c in range(C):
			centers[k, c] += data[n, c]

	for k in range(K):
		if sizes[k] == 0:
			continue
		for c in range(C):
			centers[k, c] /= sizes[k]

	return True


cdef bool find_split(float[:, ::1] data, int[::1] clusters, int K, int nsplit):
	cdef int N = data.shape[0]
	cdef int C = data.shape[1]
	cdef int M

	cdef vector[vector[int]] inv_centroids

	cdef int i, c, k, n, j
	cdef float var, curr_var = 0
	cdef int curr_axis, curr_center
	cdef float[::1] axvar1 = np.zeros(C, np.float32)
	cdef float[::1] axvar2 = np.zeros(C, np.float32)

	# Init structure
	for k in range(K):
		inv_centroids.push_back(vector[int]())

	# Initialize mapping
	for i in range(N):
		k = clusters[i]
		inv_centroids[k].push_back(i)

	# Find axis of cluster with higher variance
	for k in range(K):
		M = inv_centroids[k].size()
		if M == 0:
			continue
		axvar1[:] = 0; axvar2[:] = 0
		for i in range(M):
			n = inv_centroids[k][i]
			for c in range(C):
				axvar1[c] += data[n, c]
				axvar2[c] += data[n, c] ** 2
		for c in range(C):
			var = axvar2[c] / M - (axvar1[c] / M)**2
			if var > curr_var:
				curr_var = var
				curr_center = k
				curr_axis = c

	# Find optimal split
	cdef float min_cost = np.inf
	cdef float curr_cost, min_split = 0
	cdef float curr, tmp
	cdef float left, sqleft, right, sqright, varleft, varright
	cdef int nleft, nright

	if curr_var < 1e-7:
		return False

	M = inv_centroids[curr_center].size()
	if M < nsplit:
		nsplit = M

	cdef float vmin = np.inf
	cdef float vmax = -np.inf

	for i in range(M):
		n = inv_centroids[curr_center][i]
		curr = data[n, curr_axis]
		if curr < vmin:
			vmin = curr
		if curr > vmax:
			vmax = curr

	cdef float shift = (vmax - vmin ) / nsplit

	for i in range(nsplit-1):
		curr = vmin + shift * (i + 1)
		left = 0; sqleft = 0; nleft = 0;
		right = 0; sqright = 0; nright = 0;

		for j in range(M):
			n = inv_centroids[curr_center][j]
			tmp = data[n, curr_axis]
			if tmp < curr:
				left += tmp; sqleft += tmp * tmp; nleft += 1
			else:
				right += tmp; sqright += tmp * tmp; nright += 1;

		if nleft == 0 or nright == 0:
			continue

		varleft = sqleft / nleft - (left / nleft)**2
		varright = sqright / nright - (right / nright)**2
		curr_cost = varleft + varright
		if curr_cost < min_cost:
			min_cost = curr_cost
			min_split = curr

	# Reasign clusters
	for i in range(M):
		n = inv_centroids[curr_center][i]
		if data[n, curr_axis] < min_split:
			clusters[n] = K

	return True


def kmeans_iter(np.ndarray[float, ndim=2, mode='c'] data,
				int K=2, int niter=5):
	cdef int N = data.shape[0]
	cdef int C = data.shape[1]
	cdef int[::1] counts = np.zeros(K, np.int32)
	cdef int[::1] centroids = np.zeros(N, np.int32)

	# SMURFS quick initilization
	cdef float[:, ::1] centers = np.zeros((K, C), np.float32)

	cdef int n, c, k, it, center
	cdef float dist, diff, d

	# Init centroids
	cdef bool split = initial_centroids(data, centers, K)

	if not split:
		return False

	# KMEANS optimization
	for it in range(niter):

		for n in range(N):
			dist = np.inf
			center = 0

			for k in range(K):
				diff = 0
				for c in range(C):
					d = data[n, c] - centers[k, c]
					diff += (d * d)
				diff = sqrt(diff)
				if diff < dist:
					dist = diff
					center = k

			centroids[n] = center

		counts[:] = 0
		centers[:] = 0

		for n in range(N):
			center = centroids[n]
			counts[center] += 1
			for c in range(C):
				centers[center, c] += data[n, c]

		for k in range(K):
			if counts[k] == 0:
				continue
			for c in range(C):
				centers[k, c] /= counts[k]

	return np.asarray(centers)
