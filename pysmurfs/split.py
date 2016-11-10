

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from .misc import iteritems, grid_edges, Timer
from ._merge import merge_regions, relabel_regions
from ._split import extract_regions, kmeans_iter

from smaxflow import solve_mrf, solve_smurfs, solve_binary_mrf, solve_binary_smurfs


def solve(edges, unary, pairwise, label_cost, niter=-1, verbose=False):
	if unary.shape[1] > 2:
		if label_cost.ndim == 2:
			return solve_mrf(edges, unary, pairwise, label_cost, n_iter=niter, verbose=verbose)
		else:
			return solve_smurfs(edges, unary, pairwise, label_cost, n_iter=niter, verbose=verbose)
	else:
		if label_cost.ndim == 2:
			return solve_binary_mrf(edges, unary, pairwise, label_cost)
		else:
			return solve_binary_smurfs(edges, unary, pairwise, label_cost)


def sample_random(colors, K=2, **kwargs):
	idx = np.random.randint(0, colors.shape[0], size=K)
	return colors, colors[idx]


def sample_kmeans(colors, K=2, niter=-1, **kwargs):
	if niter <= 0:
		centers = MiniBatchKMeans(n_clusters=K).fit(colors).cluster_centers_
	else:
		centers = kmeans_iter(colors, K=K, niter=niter)
	return colors, centers


sampling_functions = {
	'random'        : sample_random,
	'kmeans'        : sample_kmeans,
}

def split_regions(data, segments, edges=None, lamda=1., mode='kmeans',
				  K=5, clamda=50, unarysq=False, gamma=None, cost='potts',
				  min_size=-1, max_opt_iter=3, kiter=-1, verbose=False):
	"""

	"""
	im_shape = segments.shape
	segments = segments.ravel()
	nsegment = segments.max() + 1

	regions = extract_regions(segments)

	min_size = max(min_size, K)

	# Unary and label costs
	unary = np.full((segments.size, K), np.inf, np.float32)
	if cost == 'potts':
		lcost = np.ones((K,K), np.float32)
		lcost[np.diag_indices(K)] = 0
	else:
		lcost = np.ones((segments.size, K, K), np.float32)

	for i, idx in iteritems(regions):
		if len(idx) > min_size:
			colors = data[idx]
			colors, centers = sampling_functions[mode](colors, K=K, shape=im_shape,
													   idx=idx, clamda=clamda, niter=kiter)
			# Unary
			if centers is False: # constant region
				unary[idx, 0] = 0
				continue
			else:
				unary[idx] = euclidean_distances(colors, centers, squared=unarysq)
		else:
			unary[idx, 0] = 0
			continue

		# Label Cost
		if cost == 'l1':
			lcost[i] = manhattan_distances(centers)
		elif cost == 'l2':
			lcost[i] = euclidean_distances(centers)

	# Edges
	if edges is None:
		edges = grid_edges(im_shape)

	# Filter edges of pixels in the same segment
	idx = segments[edges[:, 0]] == segments[edges[:, 1]]
	edges = edges[idx].copy()

	if lamda < 1e-7:
		result = unary.argmin(1).astype(np.int32)
	else:
		# Pairwise costs
		diff = np.sum((data[edges[:, 0]] - data[edges[:, 1]])**2, axis=1)
		if gamma is None:
			pairwise = 1. / (1. + diff)
		elif gamma == 'ones':
			pairwise = np.ones(diff.shape[0], np.float32)
		else:
			pairwise = np.exp(-gamma * diff)
		result = solve(edges, unary, lamda * pairwise, lcost, niter=max_opt_iter, verbose=verbose)

	# Relabel
	result = relabel_regions(segments, result, edges).reshape(im_shape)

	return result
