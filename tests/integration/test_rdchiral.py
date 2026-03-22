import json
import os

import pytest
from rdkit import Chem

from rdchiral.main import (
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
    rdchiralRunText,
)


def _normalize_smiles_list(smiles_list):
    normalized = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"Invalid SMILES: {s!r}")
        normalized.append(Chem.MolToSmiles(m, isomericSmiles=True, canonical=True))
    return normalized


def _load_rdchiral_cases():
    with open(
        os.path.join(os.path.dirname(__file__), "test_rdchiral_cases.json"), "r"
    ) as fid:
        return json.load(fid)


_RDCHIRAL_CASES = _load_rdchiral_cases()


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_case",
    _RDCHIRAL_CASES,
    ids=[f"case_{i}" for i in range(len(_RDCHIRAL_CASES))],
)
def test_rdchiral_case(test_case):
    reaction_smarts = test_case["smarts"]
    reactant_smiles = test_case["smiles"]
    expected = test_case["expected"]
    expected_norm = _normalize_smiles_list(expected)

    outcomes_from_text = rdchiralRunText(reaction_smarts, reactant_smiles)
    assert _normalize_smiles_list(outcomes_from_text) == expected_norm

    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    for _ in range(3):
        outcomes_from_init = rdchiralRun(rxn, reactants)
        assert _normalize_smiles_list(outcomes_from_init) == expected_norm
