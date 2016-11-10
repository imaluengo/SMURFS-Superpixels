#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: language=c++

import numpy as np
cimport numpy as np
from cpython cimport bool

from cython.operator cimport dereference as deref, preincrement as inc

from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
from libc.math cimport sqrt


cdef inline float fmin(float a, float b): return a if a < b else b

# Disjoint set Structure
cdef int find_root(int[::1] mapping, int p):
	cdef int q = p
	while mapping[q] != q:
		q = mapping[q]
	mapping[p] = q # quick lookup
	return q


cdef int join_trees(int[::1] mapping, int[::1] sizes, int p, int q):
	cdef int r
	if sizes[p] > sizes[q]:
		mapping[q] = p
		r = p
	else:
		mapping[p] = q
		r = q
	sizes[r] = sizes[p] + sizes[q]
	return r


cdef void remap(const int[::1] mapping, int[::1] labels):
	cdef int N = mapping.shape[0]
	cdef int p, n
	cdef int curr_label = 0, new_label

	for n in range(N):
		p = find_root(mapping, n)
		if labels[p] < 0:
			new_label = curr_label
			labels[p] = new_label
			curr_label += 1
		else:
			new_label = labels[p]

		labels[n] = new_label


def relabel_regions(int[::1] segments, int[::1] regions, int[:, ::1] edges):
	cdef int nsp = regions.shape[0]
	cdef int N = regions.shape[0]
	cdef int[::1] mapping = np.arange(N, dtype=np.int32)
	cdef int[::1] sizes = np.ones(N, dtype=np.int32)

	# Connected components
	cdef int n, E = edges.shape[0]
	cdef int p, q, z, e0, e1

	# Remapping
	cdef int[::1] labels = np.full(N, -1, dtype=np.int32)
	cdef int curr_label = 0, new_label

	# Find connected components
	for n in range(E):
		e0 = edges[n, 0]
		e1 = edges[n, 1]
		if regions[e0] != regions[e1]:
			continue
		p = find_root(mapping, e0)
		q = find_root(mapping, e1)
		if p != q:
			join_trees(mapping, sizes, p, q)
			nsp -= 1

	remap(mapping, labels)

	return np.asarray(labels)


cdef cppclass Edge:
	int e1, e2
	float w

	bint lessthan "operator<"(const Edge& t) const:
		return this.w < t.w


def merge_regions(float[:, ::1] data, int[::1] regions, int[:, ::1] edges,
				  float scale=-1, bool exact=False, int min_size=-1,
				  int min_sp=-1, bool verbose=False):

	cdef int N = data.shape[0]
	cdef int K = data.shape[1]
	cdef int E = edges.shape[0]
	cdef int nsp = np.max(regions) + 1
	cdef int S = 0
	cdef int n, k, e, e0, e1, r, r0, r1, size
	cdef float cost, capp, csize
	cdef tuple t

	cdef set pairs = set()

	cdef float[:, ::1] spcolor = np.zeros((nsp, K), np.float32)
	cdef float[::1] spcost = np.zeros(nsp, np.float32)
	cdef int[::1] spsizes = np.zeros(nsp, np.int32)

	# max possible edges
	cdef vector[Edge] spedges
	cdef vector[Edge].iterator it
	cdef Edge spedge

	# Parent structure
	cdef np.ndarray[int, ndim=1, mode='c'] mapping = np.arange(nsp, dtype=np.int32)
	cdef np.ndarray[int, ndim=1, mode='c'] labels = np.full(nsp, -1, np.int32)
	cdef int curr_label = 0, new_label

	# Color features
	for n in range(N):
		r = regions[n]
		spsizes[r] += 1
		for k in range(K):
			spcolor[r, k] += data[n, k]

	for n in range(nsp):
		size = spsizes[n]
		if size == 0:
			continue
		for k in range(K):
			spcolor[n, k] /= size

	# Superpixel edges
	for e in range(E):
		e0 = edges[e, 0]
		e1 = edges[e, 1]

		r0 = regions[e0]
		r1 = regions[e1]
		if r0 != r1:
			t = (r1, r0) if r1 > r0 else (r0, r1)
			if not t in pairs:
				# Appearance cost
				cost = 0
				for k in range(K):
					cost += (spcolor[r0, k] - spcolor[r1, k])**2

				spedge.e1 = r0
				spedge.e2 = r1
				spedge.w = cost

				spedges.push_back(spedge)

				# Add existing edge
				pairs |= set([t])

	# sort edges
	stdsort(spedges.begin(), spedges.end())

	S = spedges.size()

	curr_nsp = nsp

	if verbose:
		print "-- INITIAL:", curr_nsp

	# MSF MERGING
	if scale > 0:
		it = spedges.begin()
		while it != spedges.end() and curr_nsp > min_sp:
			p = find_root(mapping, deref(it).e1)
			q = find_root(mapping, deref(it).e2)
			if p == q:
				inc(it)
				continue
			icost1 = spcost[p] + scale / spsizes[p]
			icost2 = spcost[q] + scale / spsizes[q]
			w = deref(it).w
			if w < fmin(icost1, icost2):
				z = join_trees(mapping, spsizes, p, q)
				spcost[z] = w
				curr_nsp -= 1
				it = spedges.erase(it)
			else:
				inc(it)

		if verbose:
			print "-- MSF:", curr_nsp

	if min_size > 0 and curr_nsp > min_sp:
		it = spedges.begin()
		while it != spedges.end() and curr_nsp > min_sp:
			p = find_root(mapping, deref(it).e1)
			q = find_root(mapping, deref(it).e2)
			if p != q and (spsizes[p] < min_size or spsizes[q] < min_size):
				z = join_trees(mapping, spsizes, p, q)
				spcost[z] = spedges[n].w
				curr_nsp -= 1
				it = spedges.erase(it)
			else:
				inc(it)

		if verbose:
			print "-- SMALL:", curr_nsp

	if exact:
		while curr_nsp > min_sp:
			min_size *= 2
			it = spedges.begin()
			while it != spedges.end() and curr_nsp > min_sp:
				p = find_root(mapping, deref(it).e1)
				q = find_root(mapping, deref(it).e2)
				if p != q and (spsizes[p] < min_size or spsizes[q] < min_size):
					z = join_trees(mapping, spsizes, p, q)
					spcost[z] = spedges[n].w
					curr_nsp -= 1
					it = spedges.erase(it)
				else:
					inc(it)
		if verbose:
			print "-- EXACT:", curr_nsp

	# REMAP
	remap(mapping, labels)

	return labels[regions]
