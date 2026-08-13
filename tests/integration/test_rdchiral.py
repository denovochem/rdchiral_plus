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
    max_depth = test_case.get("max_depth", 1)
    expected = test_case["expected"]
    expected_norm = _normalize_smiles_list(expected)

    outcomes_from_text = rdchiralRunText(
        reaction_smarts, reactant_smiles, max_depth=max_depth
    )
    assert _normalize_smiles_list(outcomes_from_text) == expected_norm

    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    for _ in range(3):
        outcomes_from_init = rdchiralRun(rxn, reactants, max_depth=max_depth)
        assert _normalize_smiles_list(outcomes_from_init) == expected_norm


def test_rdchiralRun_multi_depth_repeated_runs_are_identical():
    """Repeated multi-depth runs with the same rxn object must produce identical results.

    This is a regression test for a bug where reset() assigned template
    originals by reference instead of creating copies. After the first run,
    template atom-map numbers were corrupted, causing subsequent runs to
    produce different results.
    """
    smarts = "[c:1]-[N;H0;D3;+1:2](-[O;H0;D1;-1])=[O;H0;D1;+0]>>[c;+0:1]-[N;H2;D1;+0:2]"
    smiles = "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"

    rxn = rdchiralReaction(smarts)
    reactants = rdchiralReactants(smiles)

    results = []
    for _ in range(3):
        outcomes, mapped = rdchiralRun(rxn, reactants, max_depth=3, return_mapped=True)
        results.append((_normalize_smiles_list(outcomes), mapped))

    # All three runs should produce identical outcomes
    for i in range(1, len(results)):
        assert results[i][0] == results[0][0], (
            f"Run {i} outcomes differ from run 0: {results[i][0]} vs {results[0][0]}"
        )
        assert results[i][1] == results[0][1], f"Run {i} mapped dict differs from run 0"
