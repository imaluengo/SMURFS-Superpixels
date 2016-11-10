

import numpy as np
from math import sqrt, floor

from .split import split_regions
from ._merge import merge_regions
from .misc import preprocess, grid_edges, color_gmagnitude, Timer


def rSMURFS(img, num_superpixels=200, xylamda=10, **kwargs):
	"""Regular SMURFS"""
	return SMURFS(img, num_superpixels=num_superpixels,
				  xylamda=xylamda, **kwargs)


def qrSMURFS(img, num_superpixels=200, xylamda=10, **kwargs):
	"""Quick Regular SMURFS"""
	kwargs['num_iter'] = 1
	return SMURFS(img, num_superpixels=num_superpixels,
				  xylamda=xylamda, **kwargs)


def qSMURFS(img, num_superpixels=200, **kwargs):
	"""Quick SMURFS"""
	kwargs['num_iter'] = 1
	return SMURFS(img, num_superpixels=num_superpixels, **kwargs)


def SMURFS(img, num_superpixels=200, lamda=1., umode='kmeans', K=5, mu1=2,
		   small_factor=0.1, scale_factor=0.1, gamma=None, cost='potts',
		   exact=True, num_iter=None, max_opt_iter=2, kiter=5, xylamda=None,
		   return_history=False, verbose=False):
	"""
	SMURFS: Superpixels from Multiscale Refinement of Super-regions

	Main Args:
		img: ndarray (H, W, 3)
			RGB image
		num_superpixels: int
			Number of desired superpixels.
		lamda: float
			Spatial regularization constant for the split step.
		umode: 'random' or 'kmeans'
			Centroid sampling function for the split step.
		K: int
			Number of centroids sampled per region in the split step.
	Optional Args: result in variations of SMURFS
		mu1: number
			Size factor for the initial region grid:
				`num_regions = num_superpixels / mu1`
		small_factor: float
			Size factor for the minimum permited region size:
				`small_region_size <= avg_region_size * small_factor`
		scale_factor: float
			Size factor for the merging process. Larger numbers will merge more reigons.
				`scale = avg_region_size * scale_factor`
		gamma: `None`, 'auto' or float
			Constant for regularizing edge differences.
			If `None`:
				`pairwise pottential = 1. / (1. + diff)
			Else:
				If 'auto', `gamma` is empirically approximated as `gamma = mean(diff)`
				`pairwise_potential = np.exp(-gamma * diff)`
		cost: 'potts', 'l1' or 'l2'
			Label cost potential for the split step.
		exact: bool
			Wether to return (or try to) the exact number of superpixels.
		num_iter: `None` or int
			If `None` 5 will be selected if `umode = 'kmeans'` else 10
		max_opt_iter: int
			Number of alpha-expansion iterations in the split. Algorithm converges
			within 2-3 iterations.
		kiter: int
			Number of k-means iterations in the when extracting centroids
			(if `umode = 'kmeans'`). k-means is deterministically initialized
			through hierarchical clustering, thus, a very few iterations of k-means
			are needed for convergence.
		xylamda: `None` or float
			Wether to use spatial information when sampling centroids.
		verbose: bool
			Output information.
	Return:
		result: numpy (H, W) array
			Superpixel results as a label array.
	"""
	if num_iter is None:
		num_iter = 5 if umode.startswith('kmeans') else 10

	dshape = img.shape[:2]
	avg_size = np.prod(dshape) // num_superpixels
	min_size_split = int(avg_size // 2)
	min_size = int(avg_size * small_factor)
	scale = float(avg_size * scale_factor)
	num_ss = int(num_superpixels // 2) if not exact else int(num_superpixels)

	data = preprocess(img)

	if xylamda is not None:
		y, x = np.mgrid[:dshape[0], :dshape[1]]
		y = y.ravel().astype(np.float32) / max(dshape)
		x = x.ravel().astype(np.float32) / max(dshape)
		data = np.c_[y * xylamda, x * xylamda, data]

	if return_history:
		sp_hist = []

	ss = initial_grid(dshape, num_superpixels, mu1=mu1)
	edges = grid_edges(dshape)

	if gamma == 'auto':
		diff = np.sum((data[edges[:, 0]] - data[edges[:, 1]])**2, axis=1)
		gamma = diff.mean()
		if verbose:
			print "[auto] Gamma:", gamma

	for i in range(num_iter-1):
		sp = split_regions(data, ss, edges=edges, lamda=lamda, gamma=gamma, kiter=kiter,
						   mode=umode, cost=cost, K=K, min_size=min_size_split,
						   max_opt_iter=max_opt_iter, verbose=verbose)
		if return_history:
			sp_hist.append(sp)

		ss = merge_regions(data, sp.ravel(), edges, scale=scale, min_size=-1,
						   min_sp=num_ss, exact=False, verbose=verbose).reshape(dshape)

	sp = split_regions(data, ss, edges=edges, lamda=lamda, gamma=gamma, kiter=kiter,
					   mode=umode, cost=cost, K=K, min_size=min_size_split,
					   max_opt_iter=max_opt_iter, verbose=verbose)

	if return_history:
		sp_hist.append(sp)

	# Merge similar + small
	sp = merge_regions(data, sp.ravel(), edges, min_size=min_size, scale=scale,
					   min_sp=num_superpixels, exact=exact, verbose=verbose).reshape(dshape)

	if return_history:
		return (sp, sp_hist)

	return sp


def initial_grid(im_shape, nsp, mu1=4., center=False):
	"""
	Creates the initial uniform grid from which supervoxels are going
	to be extracted.

	Args:
		im_shape: tuple (H, W)
			The shape of the original image.
		nsp: int
			Number of desired superpixels.
		factor: float
			Scales the grid to larger regions from the initial `nsp` estimation.
		center: bool
			Wether to center the grid or not.
	Return:
		result: numpy (H, W) array
			Initial partition.
	"""
	height, width = im_shape[:2]
	npart = int(floor(sqrt(nsp / mu1)))
	if center:
		size_y = height / float(npart - 1)
		size_x = width / float(npart - 1)
		shift_y = (height - size_y * npart) // 2
		shift_x = (width - size_x * npart) // 2
	else:
		size_y = height / float(npart)
		size_x = width / float(npart)
		shift_y = 0
		shift_x = 0

	# creates Y and X coordinate grid
	gy, gx = np.mgrid[:height, :width]
	idx_y = np.clip((gy - shift_y) // size_y, 0, npart - 1)
	idx_x = np.clip((gx - shift_x) // size_x, 0, npart - 1)

	# Assign to each pixel a superpixel depending on its position
	result = (idx_y * npart + idx_x).astype(np.int32)
	return result
