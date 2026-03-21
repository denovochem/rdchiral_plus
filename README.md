[![PyPI version](https://badge.fury.io/py/rdchiral.svg)](https://badge.fury.io/py/rdchiral)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/denovochem/rdchiral_plus/graphs/commit-activity)
[![License](https://img.shields.io/pypi/l/rdchiral)](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE)
[![Run Tests](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml/badge.svg)](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml)
[![Build Docs](https://github.com/denovochem/cholla_chem/actions/workflows/docs.yml/badge.svg)](https://github.com/denovochem/cholla_chem/actions/workflows/docs.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/denovochem/rdchiral_mypyc/blob/main/examples/example_notebook.ipynb)

# rdchiral_plus
Wrapper for RDKit's RunReactants to improve stereochemistry handling

This repository is a fork of [rdchiral](https://github.com/connorcoley/rdchiral). It has been modified for improved performance, and is statically typed wherever possible so that it can be compiled with [mypyc](https://mypyc.readthedocs.io/en/latest/introduction.html) for faster execution. These modifications provide comparable speed to the fast (5-10x) C++ version ([rdchiral_cpp](https://gitlab.com/ljn917/rdchiral_cpp)), with all of the benefits of being written in Python. It is also pip installable and cross platform.

## Requirements

* RDKit (version >= 2019)
* Python (version >= 3.5)

## Installation

Install RDChiral with pip directly from this repo:

```shell
pip install git+https://github.com/denovochem/rdchiral_plus.git
```

## Documentation

See ```rdchiral/main.py``` for a brief description of expected behavior and a few basic examples of how to use the wrapper. 

See ```rdchiral/test/test_rdchiral.py``` for a small set of test cases described [here](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00286)
