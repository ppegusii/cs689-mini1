# Word Analogy Solver
## Building
### Dependencies
This program is dependent upon Python 2.7.10, pandas 0.16.2, and numpy 1.9.3.
### Create an environment if necessary
Using conda for package management. [Here](http://conda.pydata.org/docs/using/envs.html) is a basic tutorial.
#### Create environment from file
`conda env create -f environment.yml`
#### Update environment from file
`conda env update -f environment.yml`
#### Activate the environment
Linux, OS X: `source activate mini1`

Windows: `activate mini1`
#### Deactivate the environment
Linux, OS X: `source deactivate mini1` or `source deactivate`

Windows: `deactivate mini1`
## Running
```
$ ./main.py -h
usage: main.py [-h] [-d DISTANCE_MEASURE] embeddings analogies output

Word analogy solver. Written in Python 2.7.

positional arguments:
	embeddings            The word embeddings file.
	analogies             The directory containing analogy files.
	output                The directory for output files.

optional arguments:
	-h, --help            show this help message and exit
	-d DISTANCE_MEASURE, --distance_measure DISTANCE_MEASURE
						The distance measure to use. {"cosadd", "cosmult"}
						(default: cosadd)
```
### Example
`./main.py glove.6B.50d.txt google_analogy_directory/ output_directory/ -d cosmult`
## Evaluation
`./evaluate.py outputdirectory/`
