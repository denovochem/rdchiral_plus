"""Unit tests for product-side SMARTS constraint extraction and filtering."""

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.smarts_constraints import (
    extract_product_smarts_constraints,
    filter_outcomes_by_smarts_constraints,
)

# ---------------------------------------------------------------------------
# extract_product_smarts_constraints
# ---------------------------------------------------------------------------


def test_extract_no_constraints_returns_empty_list():
    """Template with only bare element types and map numbers → empty list."""
    rxn = rdChemReactions.ReactionFromSmarts("[#6:1]-[#8:2]>>[#6:1].[#8:2]")
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    assert patterns == []


def test_extract_recursive_smarts_detected():
    """Template with !$(C(=O)O) on product atom → pattern extracted."""
    rxn = rdChemReactions.ReactionFromSmarts(
        "[#6:2]-[N:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[#8]"
    )
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    mapnums = {p[1] for p in patterns}
    assert 5 in mapnums


def test_extract_non_recursive_constraints_detected():
    """Template with H/D constraints (no recursive) on product atom → pattern extracted."""
    rxn = rdChemReactions.ReactionFromSmarts("[#6:1]-[#8:2]>>[#6;H0;D3:1].[#8;H1;D1:2]")
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    mapnums = {p[1] for p in patterns}
    assert mapnums == {1, 2}


def test_extract_multiple_recursive_on_same_atom():
    """Multiple recursive expressions on same atom → single pattern."""
    rxn = rdChemReactions.ReactionFromSmarts("[#6:1]>>[#6;!$(C(=O)O);!$(C(=O)N):1]")
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    assert len(patterns) == 1
    _, mapnum, patt = patterns[0]
    assert mapnum == 1
    assert patt is not None


def test_extract_unmapped_atoms_skipped():
    """Atoms with mapnum=0 are skipped."""
    rxn = rdChemReactions.ReactionFromSmarts("[#6:1]-[#8]>>[#6;!$(C(=O)O):1].[#8]")
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    mapnums = {p[1] for p in patterns}
    assert mapnums == {1}


def test_extract_multiple_product_fragments():
    """Recursive SMARTS on different product fragments → patterns for each."""
    rxn = rdChemReactions.ReactionFromSmarts(
        "[#6:1]-[#6:2]>>[#6;!$(C(=O)O):1].[#6;!$(C(=O)N):2]"
    )
    rxn.Initialize()
    patterns = extract_product_smarts_constraints(rxn)
    assert len(patterns) == 2
    template_indices = {p[0] for p in patterns}
    mapnums = {p[1] for p in patterns}
    assert template_indices == {0, 1}
    assert mapnums == {1, 2}


# ---------------------------------------------------------------------------
# filter_outcomes_by_smarts_constraints
# ---------------------------------------------------------------------------


def test_filter_empty_patterns_passthrough():
    """Empty patterns list → outcomes returned unchanged."""
    mol = Chem.MolFromSmiles("CC")
    outcomes = ((mol,),)
    result = filter_outcomes_by_smarts_constraints(outcomes, [])
    assert result == outcomes


def test_filter_recursive_violation_removed():
    """Outcomes where atom violates !$(C(=O)O) are filtered out."""
    # Use template without mapnum on O to avoid pre-existing KeyError: 900
    rxn_smarts = "[#6:2]-[N;H1;D2:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[O]"
    reactant_smiles = (
        "COC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)OC(C)(C)C"
    )
    mol = Chem.MolFromSmiles(reactant_smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxn.Initialize()
    outcomes = rxn.RunReactants((mol,))
    assert len(outcomes) > 1

    patterns = extract_product_smarts_constraints(rxn)
    # Filter to only the recursive pattern (mapnum 5) for this test
    recursive_patterns = [p for p in patterns if p[1] == 5]
    assert len(recursive_patterns) == 1

    filtered = filter_outcomes_by_smarts_constraints(outcomes, recursive_patterns)
    # Some outcomes should be filtered (those where atom 5 is C(=O)O)
    assert len(filtered) < len(outcomes)
    # Verify filtered outcomes don't contain C(=O)O on atom 5
    acid_patt = Chem.MolFromSmarts("C(=O)O")
    for outcome in filtered:
        product = outcome[1]  # second fragment (index 1)
        for a in product.GetAtoms():
            if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == 5:
                matches = product.GetSubstructMatches(acid_patt)
                assert not any(a.GetIdx() in m for m in matches), (
                    "Filtered outcome still has C(=O)O on atom 5"
                )


def test_filter_preserves_valid_outcomes():
    """Valid outcomes (recursive constraint satisfied) are retained."""
    rxn_smarts = "[#6:1]-[#8:2]>>[#6;!$(C(=O)O):1].[#8:2]"
    reactant_smiles = "COCc1ccccc1"  # methyl ether, no carboxylic acid
    mol = Chem.MolFromSmiles(reactant_smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxn.Initialize()
    outcomes = rxn.RunReactants((mol,))
    assert len(outcomes) > 0

    patterns = extract_product_smarts_constraints(rxn)
    # Only check the recursive pattern
    recursive_patterns = [p for p in patterns if "$" in Chem.MolToSmarts(p[2])]
    filtered = filter_outcomes_by_smarts_constraints(outcomes, recursive_patterns)
    assert len(filtered) == len(outcomes)


def test_filter_non_recursive_constraint_mismatch():
    """Strict mode catches degree mismatches (bond-breaking changes degree)."""
    # Template says D2 on product atom, but after C-O break the atom has D1
    rxn_smarts = "[#6;D2:1]-[#8;D2:2]>>[#6;!$(C(=O)O);D2:1].[#8;D2:2]"
    reactant_smiles = "COCc1ccccc1"
    mol = Chem.MolFromSmiles(reactant_smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxn.Initialize()
    outcomes = rxn.RunReactants((mol,))
    assert len(outcomes) > 0

    patterns = extract_product_smarts_constraints(rxn)
    # The pattern should include D2 constraint
    assert len(patterns) >= 1

    filtered = filter_outcomes_by_smarts_constraints(outcomes, patterns)
    # The methyl carbon after breaking has degree 1, but template says D2
    # Strict mode should filter it out
    assert len(filtered) < len(outcomes)


# ---------------------------------------------------------------------------
# Integration with rdchiralReaction / rdchiralRun
# ---------------------------------------------------------------------------


def test_rdchiralReaction_product_smarts_constraints_property():
    """rdchiralReaction exposes pre-computed product_smarts_constraints."""
    rxn = rdchiralReaction("[#6:2]-[N:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[#8]")
    patterns = rxn.product_smarts_constraints
    mapnums = {p[1] for p in patterns}
    assert 5 in mapnums


def test_rdchiralReaction_no_constraints_empty_list():
    """rdchiralReaction with simple template (no H/D/recursive) → empty constraints."""
    rxn = rdchiralReaction("[#6:1]-[#35:2]>>[#6:1].[#35:2]")
    assert rxn.product_smarts_constraints == []


def test_rdchiralRunText_default_no_filtering():
    """Without enforce_reactants_smarts_constraints, all outcomes returned."""
    # Use template without mapnum on O to avoid pre-existing KeyError: 900
    rxn_smarts = "[#6:2]-[N;H1;D2:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[O]"
    reactant_smiles = (
        "COC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)OC(C)(C)C"
    )
    outcomes_default = rdchiralRunText(rxn_smarts, reactant_smiles)
    outcomes_enforced = rdchiralRunText(
        rxn_smarts, reactant_smiles, enforce_reactants_smarts_constraints=True
    )
    # With enforcement, some invalid outcomes should be filtered
    assert len(outcomes_enforced) < len(outcomes_default)
    assert len(outcomes_enforced) > 0


def test_rdchiralRun_enforce_recursive_smarts_filters():
    """rdchiralRun with enforce_reactants_smarts_constraints filters invalid outcomes."""
    rxn_smarts = "[#6:2]-[N;H1;D2:3]-[#6:5]>>[#6:2]-[N:3].[#6;!$(C(=O)O):5]-[O]"
    reactant_smiles = (
        "COC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)OC(C)(C)C"
    )
    rxn = rdchiralReaction(rxn_smarts)
    reactants = rdchiralReactants(reactant_smiles)

    outcomes_default = rdchiralRun(rxn, reactants)
    outcomes_enforced = rdchiralRun(
        rxn, reactants, enforce_reactants_smarts_constraints=True
    )
    assert len(outcomes_enforced) < len(outcomes_default)


def test_rdchiralRun_enforce_no_recursive_no_overhead():
    """Template without recursive SMARTS → same results with/without enforcement."""
    rxn_smarts = "[#6:1]-[#35:2]>>[#6:1].[#35:2]"
    reactant_smiles = "Cc1ccccc1"
    outcomes_default = rdchiralRunText(rxn_smarts, reactant_smiles)
    outcomes_enforced = rdchiralRunText(
        rxn_smarts, reactant_smiles, enforce_reactants_smarts_constraints=True
    )
    assert sorted(outcomes_default) == sorted(outcomes_enforced)


# ---------------------------------------------------------------------------
# Reactant-side recursive SMARTS (RDKit handles these correctly)
# ---------------------------------------------------------------------------


def test_reactant_side_recursive_smarts_respected_by_runreactants():
    """RDKit's RunReactants correctly enforces reactant-side recursive SMARTS.

    This test verifies the assumption that reactant-side recursive SMARTS are
    handled by RunReactants (unlike product-side recursive SMARTS). If this
    test fails, it indicates RDKit changed behavior and we may need to add
    reactant-side filtering too.
    """
    # Reactant template: C must NOT be C(=O)O
    rxn_smarts = "[#6;!$(C(=O)O):1]-[#8:2]>>[#6:1].[#8:2]"
    # This reactant has a C-O bond in a carboxylic acid (C(=O)O)
    # and a C-O bond in an ether. The recursive constraint should prevent
    # matching the carboxylic acid C.
    reactant_smiles = "OCC(=O)O"  # HO-CH2-C(=O)-OH
    mol = Chem.MolFromSmiles(reactant_smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxn.Initialize()
    outcomes = rxn.RunReactants((mol,))

    # If reactant-side recursive is respected, the carboxylic acid C
    # should NOT be matched. The ether C (CH2) should match.
    # We expect outcomes, but none where atom 1 is the carboxylic acid C.
    assert len(outcomes) > 0
    acid_patt = Chem.MolFromSmarts("C(=O)O")
    for outcome in outcomes:
        product = outcome[0]
        for a in product.GetAtoms():
            if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == 1:
                matches = product.GetSubstructMatches(acid_patt)
                # The atom should NOT be part of a C(=O)O group
                assert not any(a.GetIdx() in m for m in matches), (
                    "Reactant-side recursive SMARTS was not respected by RunReactants"
                )


def test_filter_ring_constraint_no_ringinfo_error():
    """Ring-related SMARTS constraints (R, r) must not crash with RingInfo not initialized.

    Product molecules from RunReactants may not have RingInfo initialized.
    The filter must call FastFindRings before GetSubstructMatches to avoid
    a precondition violation.
    """
    rxn_smarts = "[#6:1]-[#8:2]>>[#6:1].[#8;R:2]"
    reactant_smiles = "C1CCOC1"
    outcomes = rdchiralRunText(
        rxn_smarts, reactant_smiles, enforce_reactants_smarts_constraints=True
    )
    # Should not raise; outcome where O is no longer in a ring is valid
    assert len(outcomes) > 0
