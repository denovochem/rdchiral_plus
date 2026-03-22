[![PyPI version](https://badge.fury.io/py/rdchiral-plus.svg)](https://badge.fury.io/py/rdchiral-plus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/denovochem/rdchiral_plus/graphs/commit-activity)
[![License](https://img.shields.io/pypi/l/rdchiral-plus)](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE)
[![Run Tests](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml/badge.svg)](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml)
[![Build Docs](https://github.com/denovochem/rdchiral_plus/actions/workflows/docs.yml/badge.svg)](https://github.com/denovochem/rdchiral_plus/actions/workflows/docs.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/denovochem/rdchiral_plus/blob/main/examples/example_notebook.ipynb)

# rdchiral_plus
Wrapper for RDKit's RunReactants to improve stereochemistry handling

This repository is a fork of [rdchiral](https://github.com/connorcoley/rdchiral). It has been modified for improved performance, and is statically typed wherever possible so that it can be compiled with [mypyc](https://mypyc.readthedocs.io/en/latest/introduction.html) for faster execution. These modifications provide comparable speed to the fast (5-10x) C++ version ([rdchiral_cpp](https://gitlab.com/ljn917/rdchiral_cpp)), with all of the benefits of being written in Python. It is also pip installable and cross platform.

## Requirements

* RDKit (version >= 2019)
* Python (version >= 3.9)

## Installation

Install rdchiral_plus from PyPI:

```bash
pip install rdchiral-plus
```

Or install rdchiral_plus with pip directly from this repo:

```bash
pip install git+https://github.com/denovochem/rdchiral_plus.git
```

For the pure python version of rdchiral_plus (no mypyc compilation), pip install from this repo and set RDCHIRAL_USE_MYPYC=0:

```bash
RDCHIRAL_USE_MYPYC=0 pip install "git+https://github.com/denovochem/rdchiral_plus.git"
```


## Basic usage
```python
from rdchiral import rdchiralRunText, rdchiralReaction, rdchiralReactants

# Run directly from SMARTS and SMILES (slower than pre-initializing rdchiralReaction and rdchiralReactants when processing a large numbers of reactions)
reaction_smarts = '[C:1][OH:2]>>[C:1][O:2][C]'
reactant_smiles = 'OCC(=O)OCCCO'
outcomes = rdchiralRunText(reaction_smarts, reactant_smiles)
print(outcomes)

# Pre-initialize then run
rxn = rdchiralReaction(reaction_smarts)
reactants = rdchiralReactants(reactant_smiles)
outcomes = rdchiralRun(rxn, reactants)
print(outcomes)

# Get list of atoms that changed
outcomes, mapped_outcomes = rdchiralRun(rxn, reactants, return_mapped=True)
print(outcomes, mapped_outcomes)
```

## Documentation
Full documentation is available [here](https://denovochem.github.io/rdchiral_plus/)

## Contributing

- Feature ideas and bug reports are welcome on the Issue Tracker.
- Fork the [source code](https://github.com/denovochem/rdchiral_plus) on GitHub, make changes and file a pull request.

## License

rdchiral_plus is licensed under the [MIT license](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE).
