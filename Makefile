all: smurfs

smurfs:
	python pysmurfs/setup.py build_ext --inplace
	python smaxflow/setup.py build_ext --inplace
