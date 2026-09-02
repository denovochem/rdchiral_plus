# rdchiral_plus
Wrapper for RDKit's RunReactants to improve stereochemistry handling

This repository is a fork of [RDChiral](https://github.com/connorcoley/rdchiral). It has been modified for improved speed and accuracy while maintaining high interface consistency with the upstream library. These modifications provide speed that is marginally slower than the fast C++ version ([rdchiral_cpp](https://gitlab.com/ljn917/rdchiral_cpp)), but has the benefits of being written in Python. This library is pip installable cross platform.

## Requirements

* RDKit (version >= 2019)
* Python (version >= 3.10)

## Installation

Install rdchiral_plus from PyPI:

```bash
pip install rdchiral-plus
```

Or install rdchiral_plus with pip directly from this repo:

```bash
pip install git+https://github.com/denovochem/rdchiral_plus.git
```

This fork can be optionally compiled with [mypyc](https://mypyc.readthedocs.io/en/latest/introduction.html). In our testing performance is not noticeably improved, as most of the computationally expensive work in this library is done with rdkit, which is already primarily written in C++.

For mypyc compilation:

```bash
RDCHIRAL_USE_MYPYC=1 pip install "git+https://github.com/denovochem/rdchiral_plus.git"
```

## Basic usage
```python
from rdchiral import rdchiralRun, rdchiralRunText, rdchiralReaction, rdchiralReactants

# Run directly from SMARTS and SMILES (slower than pre-initializing rdchiralReaction and rdchiralReactants when processing a large numbers of reactions)
reaction_smarts = "[C:1][OH:2]>>[C:1][O:2][C]"
reactant_smiles = "OCC(=O)OCCCO"
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

## Changes from upstream rdchiral

See the [README](https://github.com/denovochem/rdchiral_plus#changes-to-template-application) for a full list of changes to template application, template extraction, and general improvements.

## Consistency with the upstream library

See [consistency](consistency.md) for details on how consistency is measured against the original library and full details of what changes you can expect compared to the original RDChiral library.

## Contributing

- Feature ideas and bug reports are welcome on the Issue Tracker.
- Fork the [source code](https://github.com/denovochem/rdchiral_plus) on GitHub, make changes and file a pull request.

## License

rdchiral_plus is licensed under the [MIT license](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE).

## References

- [Original RDChiral paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00286)
- [rdchiral repo](https://github.com/connorcoley/rdchiral)
- [rdchiral_cpp repo](https://gitlab.com/ljn917/rdchiral_cpp)