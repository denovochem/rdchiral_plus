"""Integration tests for enforce_reactants_smarts_constraints filtering."""

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
    return sorted(normalized)


def _load_cases():
    with open(
        os.path.join(os.path.dirname(__file__), "test_smarts_constraints_cases.json"),
        "r",
    ) as fid:
        return json.load(fid)


_CASES = _load_cases()


@pytest.mark.parametrize(
    "test_case",
    _CASES,
    ids=[c["name"] for c in _CASES],
)
def test_recursive_smarts_integration(test_case):
    """Verify enforce_reactants_smarts_constraints produces expected outcomes."""
    reaction_smarts = test_case["smarts"]
    reactant_smiles = test_case["smiles"]
    enforce = test_case["enforce"]
    expected = test_case["expected"]
    expected_norm = _normalize_smiles_list(expected)

    # Test via rdchiralRunText
    outcomes_from_text = rdchiralRunText(
        reaction_smarts,
        reactant_smiles,
        enforce_reactants_smarts_constraints=enforce,
    )
    assert _normalize_smiles_list(outcomes_from_text) == expected_norm

    # Test via rdchiralRun with pre-initialized objects
    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    outcomes_from_init = rdchiralRun(
        rxn, reactants, enforce_reactants_smarts_constraints=enforce
    )
    assert _normalize_smiles_list(outcomes_from_init) == expected_norm


def test_enforce_repeated_runs_are_identical():
    """Repeated runs with enforcement must produce identical results."""
    smarts = "[#6:2]-[N;H1;D2:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[O]"
    smiles = "COC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)OC(C)(C)C"

    rxn = rdchiralReaction(smarts)
    reactants = rdchiralReactants(smiles)

    results = []
    for _ in range(3):
        outcomes = rdchiralRun(
            rxn, reactants, enforce_reactants_smarts_constraints=True
        )
        results.append(_normalize_smiles_list(outcomes))

    for i in range(1, len(results)):
        assert results[i] == results[0], (
            f"Run {i} outcomes differ from run 0: {results[i]} vs {results[0]}"
        )
