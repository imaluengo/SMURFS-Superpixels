

#ifndef __SMURFS_SPLIT__
#define __SMURFS_SPLIT__


#include "maxflow_src/graph.h"

#include <stdio.h>
#include <vector>       // vector
#include <float.h>      // FLT_MAX
#include <cstdlib>      // std::rand, std::srand


void mf_solve_binary_mrf(const int* edges, const float* unary_cost,
						 const float* pairwise_cost, const float* label_cost,
						 int *result,
						 int n_nodes, int n_edges);

void mf_solve_mrf(const int* edges, const float* unary_cost,
				  const float* pairwise_cost, const float* label_cost,
				  int *result,
				  int n_nodes, int n_edges, int n_labels, int n_iter,
				  unsigned long random_seed=42);


void mf_solve_binary_smurfs(const int* edges, const float* unary_cost,
							const float* pairwise_cost, const float* label_cost,
							int *result,
							int n_nodes, int n_edges,
							const int* regions, int n_regions);

void mf_solve_smurfs(const int* edges, const float* unary_cost,
					 const float* pairwise_cost, const float* label_cost,
					 int *result,
					 int n_nodes, int n_edges, int n_labels, int n_iter,
					 const int* regions, int n_regions,
					 unsigned long random_seed=42);

#endif
