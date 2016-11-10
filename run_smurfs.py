#!/usr/bin/env python

import argparse

import os
import numpy as np
from skimage import io, util
from pysmurfs import SMURFS, qSMURFS, rSMURFS, qrSMURFS, visualize

base_dir = os.path.dirname(os.path.realpath(__file__))

smurfs_desc ='SMURFS: Superpixels from Multiscale Refinement of Super-regions'

parser = argparse.ArgumentParser(description=smurfs_desc)
parser.add_argument('input_file', type=str,
					help='Input RGB image file path')
parser.add_argument('num_superpixels', type=int,
					help='Desired number of superpixels.')
parser.add_argument('--quick', dest='quick', action='store_true',
					help='Run only one iteration of the algorithm.')
parser.add_argument('--regular', dest='regular', action='store_true',
					help='Obtain regular (more square) superpixels.')
parser.add_argument('--out', type=str, default=base_dir,
					help='Output folder. If not given, result will be saved in '\
						 'the current folder.')
parser.add_argument('--plot', dest='plot', action='store_true',
					help='Show plot with results after finishing.')

args = parser.parse_args()

try:
	img = io.imread(args.input_file)
	img = util.img_as_float(img).astype(np.float32)
except Exception, e:
	raise Exception('Invalid image file: {}'.format(args.input_file))

if args.num_superpixels <= 0 or args.num_superpixels >= np.prod(img.shape[:2]):
	raise Exception('Invalid number of superpixels: {}'.format(args.num_superpixels))

if args.regular:
	if args.quick:
		result = qrSMURFS(img, args.num_superpixels)
	else:
		result = qSMURFS(img, args.num_superpixels)
else:
	if args.quick:
		result = qSMURFS(img, args.num_superpixels)
	else:
		result = SMURFS(img, args.num_superpixels)

fileId = os.path.basename(args.input_file)
fileId = fileId[:fileId.rfind('.')]
out_file = 'result-{}.png'.format(fileId)
out_file = os.path.join(args.out, out_file)

io.imsave(out_file, result.astype(np.uint16))

if args.plot:
	visualize(img, result)
