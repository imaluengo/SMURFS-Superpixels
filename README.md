# SMURFS
## Superpixels from Multi-scale Refinement of Super-regions

Python plug&play library version of SMURFS. Easy to try different functions & parameters of the algorithm with the `pysmurfs` library.

## Installation

Obtain the code:

```bash
$> git clone https://github.com/imaluengo/SMURFS.git
```

Install python dependencies:
```bash
$> pip install --upgrade numpy scikit-image scikit-learn cython matplotlib
```

Compile library:
```bash
$> make
```

## Run SMURFS

```bash
$> python run_smurfs.py --help
usage: run_smurfs.py [-h] [--quick] [--regular] [--out OUT] [--plot]
					 input_file num_superpixels

SMURFS: Superpixels from Multiscale Refinement of Super-regions

positional arguments:
  input_file       Input RGB image file path
  num_superpixels  Desired number of superpixels.

optional arguments:
  -h, --help       show this help message and exit
  --quick          Run only one iteration of the algorithm.
  --regular        Obtain regular (more square) superpixels.
  --out OUT        Output folder. If not given, result will be saved in the
				   current folder.
  --plot           Show plot with results after finishing.
```

Example runs:

- **SMURFS:** 200 superpixels

	```bash
	$> python run_smurfs.py /data/BSR/BSDS500/data/images/test/100007.jpg 200 --plot
	```

- **Quick SMURFS (qSMURFS):** 200 superpixels running a single iteration of SMURFS

	```bash
	$> python run_smurfs.py /data/BSR/BSDS500/data/images/test/100007.jpg 200 --quick --plot
	```

- **Regular SMURFS (rSMURFS):** 200 superpixels with more regular shape

	```bash
	$> python run_smurfs.py /data/BSR/BSDS500/data/images/test/100007.jpg 200 --regular --plot
	```

- **Quick Regular SMURFS (qrSMURFS):** 200 superpixels running a single iteration of rSMURFS

	```bash
	$> python run_smurfs.py /data/BSR/BSDS500/data/images/test/100007.jpg 200 --quick --regular --plot
	```

## Sample Results

- **Left:** Original image.
- **Middle:** Superpixel boundaries marked over the original image.
- **Right:** Original image reconstructed by assigning to every pixel the mean color of its superpixel. Measures how well the superpixels represent the image and how much detail can be recovered from them.

![Result 1](img/fig1.png?raw=true "Result1")
![Result 2](img/fig2.png?raw=true "Result2")

## Reference

Please, consider citing the following publication if you use SMURFS within your research:

	@inproceedings{luengo2016smurfs,
	  title={SMURFS: superpixels from multi-scale refinement of super-regions},
	  author={Luengo, Imanol and Basham, Mark and French, Andrew P},
	  booktitle={British Machine Vision Conference (BMVC)},
	  year={2016}
	}

Publication available at:

[http://www.bmva.org/bmvc/2016/papers/paper004/index.html](http://www.bmva.org/bmvc/2016/papers/paper004/index.html)

**NOTE:** This is the Python version of the above publication, resulting in the same accuracy but sightly worse speed. Fully functional and stable C++ version, as presented in the above paper, is coming soon.

## Acknowledgements

Work in collaboration between:

- [University of Nottingham -- Computer Vision Laboratory](http://cvl.cs.nott.ac.uk/)

- [Diamond Light Source](http://www.diamond.ac.uk/Home.html)
