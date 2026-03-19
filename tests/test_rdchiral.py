import json
import os

import pytest

from rdchiral.main import (
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
    rdchiralRunText,
)


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

    outcomes_from_text = rdchiralRunText(reaction_smarts, reactant_smiles)
    assert outcomes_from_text == expected

    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    for _ in range(3):
        outcomes_from_init = rdchiralRun(rxn, reactants)
        assert outcomes_from_init == expected
