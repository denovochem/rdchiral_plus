# rdchiral_plus
Wrapper for RDKit's RunReactants to improve stereochemistry handling

This repository is a fork of [rdchiral](https://github.com/connorcoley/rdchiral). It has been modified for improved performance, and is statically typed wherever possible so that it can be compiled with [mypyc](https://mypyc.readthedocs.io/en/latest/introduction.html) for faster execution while maintaining consistency with the upstream library. These modifications provide comparable speed to the fast C++ version ([rdchiral_cpp](https://gitlab.com/ljn917/rdchiral_cpp)), with all of the benefits of being written in Python. This library is pip installable and cross platform.

## Requirements

* RDKit (version >= 2019)
* Python (version >= 10)

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

## References

- [Original rdchiral paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00286)
- [rdchiral repo](https://github.com/connorcoley/rdchiral)
- [rdchiral_cpp repo](https://gitlab.com/ljn917/rdchiral_cpp)