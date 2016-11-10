#!/usr/bin/env python

import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
from Cython.Distutils import build_ext

import urllib
try:
	import urllib.request
except:
	pass

import zipfile


base_path = os.path.abspath(os.path.dirname(__file__))
source_path = os.path.join(base_path, 'maxflow_src')

def get_maxflow_source():
	if os.path.isdir(source_path):
		return
	else:
		os.mkdir(source_path)

	if hasattr(urllib, "urlretrieve"):
		urlretrieve = urllib.urlretrieve
	else:
		urlretrieve = urllib.request.urlretrieve
	urlretrieve("http://vision.csd.uwo.ca/code/maxflow-v3.01.zip",
				"maxflow-v3.01.zip")
	with zipfile.ZipFile("maxflow-v3.01.zip") as zf:
		zf.extractall(source_path)


def configuration(parent_package='', top_path=None):
	config = Configuration('smaxflow', parent_package, top_path,
						   cmdclass={'build_ext': build_ext})

	numpy_dirs = get_numpy_include_dirs()
	get_maxflow_source()

	files = ["graph.cpp", "maxflow.cpp"]
	files = [os.path.join(source_path, f) for f in files]
	files = ['_maxflow.pyx', 'smurfs_mf.cpp'] + files
	config.add_extension('_maxflow', sources=files, language='c++',
						 include_dirs=[source_path,
									   get_numpy_include_dirs()],
						 library_dirs=[source_path],
						 extra_compile_args=["-fpermissive"],
						 extra_link_args=["-fpermissive"])

	return config


if __name__ == "__main__":
	config = configuration(top_path='').todict()
	setup(**config)
