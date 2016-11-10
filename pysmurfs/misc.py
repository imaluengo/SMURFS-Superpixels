

import os
import glob

import time
import numpy as np
from skimage import io
from random import choice

import matplotlib.pyplot as plt

from skimage import color
from skimage.util import img_as_float
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from skimage.filters import sobel_h, sobel_v, gaussian

from ._features import spmeans

# Python 2 and 3 compatibility
try:
	dict.iteritems
except AttributeError:
	# Python 3
	def itervalues(d):
		return iter(d.values())
	def iteritems(d):
		return iter(d.items())
else:
	# Python 2
	def itervalues(d):
		return d.itervalues()
	def iteritems(d):
		return d.iteritems()


class Timer(object):
	def __init__(self, name='Timer'):
		self.name = name

	def __enter__(self):
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		self.tend = (time.time() - self.tstart)
		print('[%s] Elapsed: %.4f seconds' % (self.name, self.tend))


def grid_edges(im_shape):
	nodes = np.arange(np.prod(im_shape)).reshape(im_shape).astype(np.int32)
	edges_y = np.c_[nodes[1:, :].ravel(), nodes[:-1, :].ravel()]
	edges_x = np.c_[nodes[:, 1:].ravel(), nodes[:, :-1].ravel()]
	edges = np.r_[edges_y, edges_x]
	return edges


def preprocess(image, use_rgb=False, use_hsv=False, norm=True):
	"""
	Preprocesses a RGB image before extracting super-regions from it. Improves
	the quality of the super-regions by transforming to the L*a*b colorspace
	and normalizing the image.

	Args:
		image: numpy (H, W) or (H, W, 3) array
			RGB image to be preprocessed.
		use_rgb: bool
			Wether to append RGB channels to the L*a*b channels.
		use_hsv: bool
			Wether to append HSV channels to the L*a*b channels.
		norm:
			Wether to standardize individual channels.
	Result:
		result: numpy (H * W, K) array
			Where K is 3, 6 or depending on `use_rgb` and `use_hsv`. channel
			specific normalization to enhance distances.
	"""
	if image.ndim == 2 or image.shape[2] == 1:
		data = (np.squeeze(image) - image.mean()) / image.std()
		return data

	assert image.shape[2] == 3, 'Error: invalid image format'

	result = color.rgb2lab(image).reshape(-1, 3)
	if use_rgb:
		result = np.column_stack(result, image.reshape(-1, 3))
	if use_hsv:
		result = np.column_stack(result, color.rgb2hsv(data).reshape(-1, 3))

	# Standardize channels and reshape in-place
	if norm:
		result = (result - result.mean(0)) / result.std(0)

	return result.astype(np.float32)


def bsdimage(idx=None, dataset=None, path='./BSD500/data/images'):
	"""
	Load an image from BSD500 dataset.

	Args:
		idx: int, str or None
			If None a random image will be loaded. Ignored if `dataset` is None
		dataset: 'train', 'val', 'test' or None
			Dataset from which to load an image.
		path: str
			Path to the BSD500 dataset.
	Return:
		RGB image from BSD500 dataset as a *numpy array*.
	Exceptions:
		Exception: if path not found
	"""
	if idx is not None and dataset is not None:
		path = os.path.join(path, dataset, '{}.jpg'.format(idx))
	else:
		if dataset not in ['train', 'val', 'test']:
			dataset = choice(['train', 'val', 'test'])
		path = choice(glob.glob(os.path.join(path, dataset, '*.jpg')))

	if os.path.isfile(path):
		return img_as_float(io.imread(path)).astype(np.float32)
	else:
		raise Exception('Image \'{}\' not found in BSD path \'{}\''
						.format(idx, os.path.join(path, dataset)))


def boundaries(regions, thin=False):
	"""
	Extracts region boundaries from a labelled image.

	Args:
		regions: numpy (H, W) array
			Regions extracted from `image`
		thin: bool
			Whether to skeletonize or not the boundaries.
	Return:
		result: numpy (H, W) array
			Region boundaries as a binary mask.
	"""
	# +1 to ignore superpixel 0 treated as background
	result = find_boundaries(regions+1)
	if thin:
		result = skeletonize(result)
	return result


def mark_boundaries(image, regions, color=(0,0,0), thin=False):
	"""
	Mark region boundaries in a given image.

	Args:
		image: numpy (H, W) array
			RGB image
		regions: numpy (H, W) array
			Regions extracted from `image`
		color: RGB tuple from 0 to 1
			Color to mark boundaries. Default black.
		thin: bool
			Whether to skeletonize or not the boundaries.
	Return:
		result: numpy (H, W) array
			Image with region boundaries overlaid on it.
	"""
	bounds = boundaries(regions, thin=thin)
	result = image.copy()
	result[bounds] = color
	return result


def reconstruct(image, superpixels, colors=None):
	"""
	Reconstruct a given image using superpixels as representative units.

	Args:
		image: numpy (H, W) array
			RGB image
		superpixels: numpy (H, W) array
			Superpixels extracted from `image`.
		color: None or numpy array of shape (num_superpixels, 3)
			Colors to use to *paint* the pixels within a superpixel. If `None`,
			mean RGB color will be extracted for every superpixel from `image`.
	Return:
		result: numpy (H, W) array
			Image reconstructed using mean color of each superpixel.
	"""
	if colors is None:
		if image.ndim == 3:
			colors = spmeans(image.reshape(-1, image.shape[-1]),
							 superpixels.ravel())
		else: # `ndim == 2` assumed, gray scale image
			colors = spmeans(image.reshape(-1, 1), superpixels.ravel())

	result = colors[superpixels]
	return result


def color_gmagnitude(img, sigma=None, norm=True, enhance=False):
	"""

	"""
	if sigma is not None:
		img = gaussian(img, sigma=sigma, multichannel=True)

	dx = np.dstack([sobel_h(img[..., i]) for i in range(img.shape[-1])])
	dy = np.dstack([sobel_v(img[..., i]) for i in range(img.shape[-1])])

	Jx = np.sum(dx**2, axis=-1)
	Jy = np.sum(dy**2, axis=-1)
	Jxy = np.sum(dx * dy, axis=-1)

	D = np.sqrt(np.abs(Jx**2 - 2 * Jx * Jy + Jy**2 + 4 * Jxy**2))
	e1 = (Jx + Jy + D) / 2. # First eigenvalue
	magnitude = np.sqrt(e1)

	if norm:
		magnitude /= magnitude.max()

	if enhance:
		magnitude = 1 - np.exp(-magnitude**2 / magnitude.mean())

	return magnitude.astype(np.float32)


def visualize(img, regions, color=(0,0,0), colors=None, thin=False,
			  text=None, axes=None, size=18):
	"""

	"""
	if axes is None:
		plot = True
		if img.shape[1] > img.shape[0]:
			fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(6 * 3, 4))
		else:
			fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(4 * 3, 6))
	else:
		plot = False

	axes[0].imshow(img);
	axes[0].set_xticks([]), axes[0].set_yticks([])
	if text is not None:
		axes[0].set_ylabel(text, size=size)
	# Option 1
	axes[1].imshow(mark_boundaries(img, regions, color=color, thin=thin))
	axes[1].set_xticks([]), axes[1].set_yticks([])
	axes[1].set_xlabel('Num superpixels: {}'.format(regions.max()+1), size=size)
	# Option 2
	axes[2].imshow(reconstruct(img, regions, colors=colors))
	axes[2].set_xticks([]), axes[2].set_yticks([])

	if plot:
		plt.tight_layout()
		plt.show()
