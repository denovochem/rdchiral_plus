[![PyPI version](https://badge.fury.io/py/rdchiral-plus.svg)](https://badge.fury.io/py/rdchiral-plus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/denovochem/rdchiral_plus/graphs/commit-activity)
[![License](https://img.shields.io/pypi/l/rdchiral-plus)](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE)
[![Run Tests](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml/badge.svg)](https://github.com/denovochem/rdchiral_plus/actions/workflows/ci.yml)
[![Build Docs](https://github.com/denovochem/rdchiral_plus/actions/workflows/docs.yml/badge.svg)](https://github.com/denovochem/rdchiral_plus/actions/workflows/docs.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/denovochem/rdchiral_plus/blob/main/examples/example_notebook.ipynb)

# rdchiral_plus
Wrapper for RDKit's RunReactants to improve stereochemistry handling

This repository is a fork of [rdchiral](https://github.com/connorcoley/rdchiral). It has been modified for improved performance while maintaining high consistency with the upstream library. Depending on the operation, rdchiral_plus is 1.7×–5× faster than the original rdchiral, and is competitive with the fast C++ version ([rdchiral_cpp](https://gitlab.com/ljn917/rdchiral_cpp)). This library has the benefits of being written in Python and achieves a higher round-trip accuracy than either the original or the C++ library. rdchiral_plus is pip installable cross platform.

The interface (`rdchiralRun`, `rdchiralRunText`, `rdchiralReaction`, `rdchiralReactants`, `extract_from_reaction`, `extract_from_reaction_smiles`, etc.) and returned data structures remain unchanged from the original library, so existing code should work with no modifications. While behavior is mostly consistent with the original library, this fork includes several important fixes and improvements.

## Changes to template application

- **[Fix cis/trans outcomes for conjugated systems](https://github.com/connorcoley/rdchiral/pull/40)**: Fixes incorrect cis/trans outcomes for conjugated systems that could previously depend on atom numbering. In particular, when a template only specifies part of a conjugated system, the copied double-bond stereo directions may need to be reversed consistently
- **Broader stereochemistry handling**: Stereochemistry for tetrahedral centers with lone pairs is accounted for
- **One-pot reactions**: Templates defining multiple reactions on the same product are now properly handled by initializing templates with parentheses where needed
- **Recursive template application**: Templates can be recursively applied with a max_depth parameter, useful for symmetric reactions, or reactions that occur at multiple sites in a molecule

## Changes to template extraction

- **[Configurable template extraction](https://github.com/connorcoley/rdchiral/commit/78bbafaba040678b957497e7f2638e935104e3d7)**: Extends template extraction to support a configurable fragment `radius` and an option to disable matching/including “special groups” (`no_special_groups`), which can change which atoms are included in extracted fragments.
- **Deterministic template extraction**: Replaced random shuffle-based tetrahedral center correction loops with deterministic permutation parity - the old behavior could lead to inconsistent results or appear to hang in rare instances.
- **Stereochemistry tracking**: Inversions of tetrahedral centers are counted as a changed atom, and included in the extracted template
- **Spectator tracking**: Spectator molecules are included in extracted template dictionaries

## General changes

- **Automatic dependency installation**: RDKit is automatically installed as a dependency


## Consistency with the upstream library

The changes above result in minor differences in behavior compared to the original library. In most cases where behavior is different, rdchiral_plus produces the more accurate result. As an example, the table below shows the roundtrip success rate of extracting a template from an atom mapped reaction SMILES, applying that template to the product SMILES, and then recovering the expected reactant SMILES. rdchiral_plus reduces the number of incorrect roundtrips by 98% compared to rdchiral, and 99% compared to rdchiral_cpp, achieving a 99.96% roundtrip success rate for extract template -> apply to products -> recover expected reactants. 

| library | successful roundtrips | success rate |
| --- | :---: | :---: |
| rdchiral | 49223 / 50016 | 98.41% |
| rdchiral_cpp | 48694 / 50016 | 97.36% |
| rdchiral_plus | 49998 / 50016 | 99.96% |


See [here](docs/consistency.md) for details on how consistency is measured against the original library and full details of what changes you can expect compared to the original rdchiral library.

## Requirements

- RDKit (version >= 2019)
- Python (version >= 3.10)

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

# Run directly from SMARTS and SMILES strings
# This is slower than pre-initializing rdchiralReaction and rdchiralReactants when
# processing a large number of reactions
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

## Contributing

- Feature ideas and bug reports are welcome on the Issue Tracker.
- Fork the [source code](https://github.com/denovochem/rdchiral_plus) on GitHub, make changes and file a pull request.

## License

rdchiral_plus is licensed under the [MIT license](https://github.com/denovochem/rdchiral_plus/blob/main/LICENSE).

## References

- [Original rdchiral paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00286)
- [rdchiral repo](https://github.com/connorcoley/rdchiral)
- [rdchiral_cpp repo](https://gitlab.com/ljn917/rdchiral_cpp)
