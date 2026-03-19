import json
import os

import pytest

from rdchiral.main import (
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
    rdchiralRunText,
)

Chem = pytest.importorskip("rdkit.Chem")


def _load_atom_mapping_cases():
    with open(
        os.path.join(os.path.dirname(__file__), "test_atom_mapping_cases.json"), "r"
    ) as fid:
        return json.load(fid)


_ATOM_MAPPING_CASES = _load_atom_mapping_cases()


def canonicalize_outcomes(outcomes):
    """Convert all SMILES in a list of outcomes to the canonical form"""
    return list(map(lambda x: Chem.CanonSmiles(x), outcomes))


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_case",
    _ATOM_MAPPING_CASES,
    ids=[f"case_{i}" for i in range(len(_ATOM_MAPPING_CASES))],
)
def test_atom_mapping_case(test_case):
    reaction_smarts = test_case["smarts"]
    reactant_smiles = test_case["smiles"]
    reactants = rdchiralReactants(reactant_smiles, custom_reactant_mapping=True)
    expected = canonicalize_outcomes(test_case["expected"])

    outcomes_from_text = canonicalize_outcomes(
        rdchiralRunText(
            reaction_smarts,
            reactant_smiles,
            custom_reactant_mapping=True,
            keep_mapnums=True,
        )
    )
    assert outcomes_from_text == expected

    rxn = rdchiralReaction(reaction_smarts)
    for _ in range(3):
        outcomes_from_init = canonicalize_outcomes(
            rdchiralRun(rxn, reactants, keep_mapnums=True)
        )
        assert outcomes_from_init == expected
