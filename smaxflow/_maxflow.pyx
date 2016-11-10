#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

from libcpp.vector cimport vector


cdef extern from "smurfs_mf.hpp":
	void mf_solve_binary_mrf(const int* edges, const float* unary_cost,
							 const float* pairwise_cost, const float* label_cost,
							 int *result,
							 int n_nodes, int n_edges) nogil except +

	void mf_solve_mrf(const int* edges, const float* unary_cost,
					  const float* pairwise_cost, const float* label_cost,
					  int *result,
					  int n_nodes, int n_edges, int n_labels, int n_iter,
					  unsigned long random_seed) nogil except +

	void mf_solve_binary_smurfs(const int* edges, const float* unary_cost,
								const float* pairwise_cost, const float* label_cost,
								int *result,
								int n_nodes, int n_edges,
								const int* regions, int n_regions) nogil except +

	void mf_solve_smurfs(const int* edges, const float* unary_cost,
						 const float* pairwise_cost, const float* label_cost,
						 int *result,
						 int n_nodes, int n_edges, int n_labels, int n_iter,
						 const int* regions, int n_regions,
						 unsigned long random_seed) nogil except +


def solve_binary_mrf(int[:, ::1] edges, float[:, ::1] unary_cost,
					 float[::1] pairwise_cost, float[:, ::1] label_cost):

	cdef int n_nodes = unary_cost.shape[0]
	cdef int n_edges = edges.shape[0]

	# create qpbo object
	cdef float* data_ptr = <float*>&(unary_cost[0,0])
	cdef int* edge_ptr = <int*>&(edges[0,0])
	cdef float* pairwise_cost_ptr = <float*>&(pairwise_cost[0])
	cdef float* label_cost_ptr = <float*>&(label_cost[0,0])

	cdef int[::1] result = np.zeros(n_nodes, np.int32)
	cdef int *result_ptr = <int*>&(result[0])

	with nogil:
		mf_solve_binary_mrf(edge_ptr, data_ptr, pairwise_cost_ptr, label_cost_ptr,
							result_ptr, n_nodes, n_edges)

	return np.asarray(result)


def solve_mrf(int[:, ::1] edges, float[:, ::1] unary_cost,
			  float[::1] pairwise_cost, float[:, ::1] label_cost,
			  int[::1] init_labels=None, int n_iter=5,
			  bool verbose=False, long random_seed=42):

	cdef int n_nodes = unary_cost.shape[0]
	cdef int n_labels =  unary_cost.shape[1]
	cdef int n_edges = edges.shape[0]
	cdef int[::1] result

	# initial guess
	if init_labels is None:
		result = np.zeros(n_nodes, dtype=np.int32)
	else:
		result = init_labels.copy()

	cdef int* edge_ptr = <int*>&(edges[0,0])
	cdef int* result_ptr = <int*>&(result[0])
	cdef float* unary_ptr = <float*>&(unary_cost[0,0])
	cdef float* pairwise_cost_ptr = <float*>&(pairwise_cost[0])
	cdef float* label_cost_ptr = <float*>&(label_cost[0,0])

	with nogil:
		mf_solve_mrf(edge_ptr, unary_ptr, pairwise_cost_ptr, label_cost_ptr,
					 result_ptr, n_nodes, n_edges, n_labels, n_iter, random_seed)

	return np.asarray(result)


def solve_binary_smurfs(int[:, ::1] edges, float[:, ::1] unary_cost,
						float[::1] pairwise_cost, float[:, :, ::1] label_cost,
						int[::1] regions):

	cdef int n_nodes = unary_cost.shape[0]
	cdef int n_edges = edges.shape[0]
	cdef int n_region = np.max(regions)+1

	# create qpbo object
	cdef float* data_ptr = <float*>&(unary_cost[0,0])
	cdef int* edge_ptr = <int*>&(edges[0,0])
	cdef float* pairwise_cost_ptr = <float*>&(pairwise_cost[0])
	cdef float* label_cost_ptr = <float*>&(label_cost[0,0,0])
	cdef int* region_ptr = <int*>&(regions[0])

	cdef int[::1] result = np.zeros(n_nodes, np.int32)
	cdef int *result_ptr = <int*>&(result[0])

	with nogil:
		mf_solve_binary_smurfs(edge_ptr, data_ptr, pairwise_cost_ptr, label_cost_ptr,
							   result_ptr, n_nodes, n_edges, region_ptr, n_region)

	return np.asarray(result)


def solve_smurfs(int[:, ::1] edges, float[:, ::1] unary_cost,
				 float[::1] pairwise_cost, float[:, :, ::1] label_cost,
				 int[::1] regions, int[::1] init_labels=None,
				 int n_iter=5, bool verbose=False, unsigned long random_seed=42):

	cdef int n_nodes = unary_cost.shape[0]
	cdef int n_labels =  unary_cost.shape[1]
	cdef int n_edges = edges.shape[0]
	cdef int n_region = np.max(regions)+1
	cdef int[::1] result

	# initial guess
	if init_labels is None:
		result = np.zeros(n_nodes, dtype=np.int32)
	else:
		result = init_labels.copy()

	cdef int* edge_ptr = <int*>&(edges[0,0])
	cdef int* result_ptr = <int*>&(result[0])
	cdef float* unary_ptr = <float*>&(unary_cost[0,0])
	cdef float* pairwise_cost_ptr = <float*>&(pairwise_cost[0])
	cdef float* label_cost_ptr = <float*>&(label_cost[0,0,0])
	cdef int* region_ptr = <int*>&(regions[0])

	with nogil:
		mf_solve_smurfs(edge_ptr, unary_ptr, pairwise_cost_ptr, label_cost_ptr,
						result_ptr, n_nodes, n_edges, n_labels, n_iter,
						region_ptr, n_region, random_seed)

	return np.asarray(result)
