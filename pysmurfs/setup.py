#!/usr/bin/env python

import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
from Cython.Distutils import build_ext

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
	config = Configuration('pysmurfs', parent_package, top_path,
						   cmdclass={'build_ext': build_ext})

	numpy_dirs = get_numpy_include_dirs()

	# Compile `_features.pyx`
	config.add_extension('_features', sources=['_features.pyx'],
						 include_dirs=numpy_dirs)

	# Compile `_split.pyx`
	config.add_extension('_split', sources=['_split.pyx'],
						 language='c++', include_dirs=numpy_dirs)

	# Compile `_relabel.pyx`
	config.add_extension('_merge', sources=['_merge.pyx'],
						 language='c++', include_dirs=numpy_dirs)

	return config


if __name__ == "__main__":
	config = configuration(top_path='').todict()
	setup(**config)
