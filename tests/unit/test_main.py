import pytest
from rdkit import Chem

from rdchiral.main import (
    deduplicate_outcomes_with_smiles,
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
    rdchiralRunText,
)


def _maps_in_smiles(smiles: str) -> set[int]:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return {a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomMapNum()}


def test_rdchiralRunText_no_outcomes_returns_empty_list():
    rxn_smarts = "[CH3:1]Br>>[CH3:1]Cl"
    reactant_smiles = "CC"  # does not match template

    assert rdchiralRunText(rxn_smarts, reactant_smiles) == []


def test_rdchiralRunText_achiral_early_return_keep_mapnums_false_strips_maps():
    rxn_smarts = "[CH3:1][Br:2]>>[CH3:1][Cl:3]"
    reactant_smiles = "CBr"  # reactants and template are achiral

    outcomes = rdchiralRunText(
        rxn_smarts,
        reactant_smiles,
        keep_mapnums=False,
        combine_enantiomers=True,
        return_mapped=False,
    )

    assert outcomes == ["CCl"]
    assert _maps_in_smiles(outcomes[0]) == set()


def test_rdchiralRunText_achiral_early_return_keep_mapnums_true_preserves_maps_and_assigns_unmapped():
    rxn_smarts = "[CH3:1][Br:2]>>[CH3:1][Cl:3]"
    reactant_smiles = "CBr"

    outcomes = rdchiralRunText(
        rxn_smarts,
        reactant_smiles,
        keep_mapnums=True,
        combine_enantiomers=True,
        return_mapped=False,
    )

    assert len(outcomes) == 1
    maps = _maps_in_smiles(outcomes[0])
    assert 1 in maps
    # The new chlorine is unmapped by default and is assigned a 900-series map number.
    assert any(m >= 900 for m in maps)


def test_rdchiralRunText_achiral_early_return_return_mapped_true_has_empty_mapped_outcomes():
    rxn_smarts = "[CH3:1][Cl:2]>>[CH3:1].[Cl:2]"
    reactant_smiles = "CBr"

    outcomes, mapped_outcomes = rdchiralRunText(
        rxn_smarts,
        reactant_smiles,
        keep_mapnums=True,
        combine_enantiomers=True,
        return_mapped=True,
    )

    assert outcomes == []
    assert mapped_outcomes == {}


def test_deduplicate_outcomes_with_smiles_removes_duplicate_outcome_tuples():
    # Construct outcomes with duplicate product tuples (by canonical smiles) to ensure
    # deterministic de-duplication.
    m1 = Chem.MolFromSmiles("CC")
    m2 = Chem.MolFromSmiles("CC")
    assert m1 is not None and m2 is not None

    outcomes = ((m1,), (m2,))
    deduped = deduplicate_outcomes_with_smiles(outcomes)

    assert len(deduped) == 1


@pytest.mark.parametrize(
    "keep_mapnums",
    [False, True],
)
def test_rdchiralRunText_simple_substitution_produces_expected_product_smiles(
    keep_mapnums,
):
    rxn_smarts = "[CH3:1][Br:2]>>[CH3:1].[Br:2]"
    reactant_smiles = "CBr"

    outcomes = rdchiralRunText(
        rxn_smarts,
        reactant_smiles,
        keep_mapnums=keep_mapnums,
        combine_enantiomers=True,
        return_mapped=False,
    )

    assert outcomes
    if keep_mapnums:
        expected = "[BrH:2].[CH3:1]"
    else:
        expected = "Br.[CH3]"
    assert expected in {
        Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) for s in outcomes
    }


def test_rdchiralRun_max_products_enforced_across_depth_levels():
    """max_products must be enforced as a hard limit across depth levels.

    This is a regression test for a bug where the max_products check only
    broke the inner parent loop but the outer depth loop continued to the
    next level, adding more products beyond the limit.
    """
    # Nitration reduction template that can match multiple times on a
    # polysubstituted aromatic ring, producing products at multiple depths.
    smarts = "[c:1]-[N;H0;D3;+1:2](-[O;H0;D1;-1])=[O;H0;D1;+0]>>[c;+0:1]-[N;H2;D1;+0:2]"
    smiles = "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"

    rxn = rdchiralReaction(smarts)
    reactants = rdchiralReactants(smiles)

    max_products = 2
    outcomes = rdchiralRun(rxn, reactants, max_depth=3, max_products=max_products)

    assert len(outcomes) <= max_products, (
        f"Expected at most {max_products} products, got {len(outcomes)}: {outcomes}"
    )
