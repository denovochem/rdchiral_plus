import pytest
from rdkit import Chem

from rdchiral.initialization import (
    initialize_rxn_from_smarts,
    rdchiralReactants,
    rdchiralReaction,
)


def _find_atom_with_mapnum(mol: Chem.Mol, mapnum: int) -> Chem.Atom:
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == mapnum:
            return a
    raise ValueError(f"No atom with mapnum {mapnum}")


def _find_atom_by_symbol(mol: Chem.Mol, symbol: str) -> Chem.Atom:
    for a in mol.GetAtoms():
        if a.GetSymbol() == symbol:
            return a
    raise ValueError(f"No atom with symbol {symbol!r}")


def test_initialize_rxn_from_smarts_remaps_unbalanced_reactant_atom_mapnums_to_700_series():
    rxn = initialize_rxn_from_smarts("[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2]")

    reactant = next(iter(rxn.GetReactants()))
    product = next(iter(rxn.GetProducts()))

    prd_maps = {a.GetAtomMapNum() for a in product.GetAtoms() if a.GetAtomMapNum()}
    assert prd_maps == {1, 2}

    br = _find_atom_by_symbol(reactant, "Br")
    assert br.GetSymbol() == "Br"
    assert br.GetAtomMapNum() >= 700
    assert br.GetAtomMapNum() not in prd_maps


def test_rdchiralReaction_lazy_init_populates_templates_and_maps_on_access():
    rxn_smarts = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][Cl:4]"
    r = rdchiralReaction(rxn_smarts, lazy_init=True)

    assert r._template_r_orig is None
    assert r._atoms_rt_map is None

    _ = r.template_r
    assert r._template_r_orig is not None
    assert r._template_r is not None

    atoms_rt_map = r.atoms_rt_map
    assert 1 in atoms_rt_map
    assert 2 in atoms_rt_map

    assert r.atoms_rt_idx_to_map
    assert r.atoms_pt_idx_to_map


def test_rdchiralReaction_template_copy_atoms_are_distinct_from_original_atoms():
    rxn_smarts = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][Cl:4]"
    r = rdchiralReaction(rxn_smarts, lazy_init=True)

    rt_orig = r.template_r_orig
    rt_copy = r.template_r

    assert rt_orig is not rt_copy

    mapnums = sorted(
        {a.GetAtomMapNum() for a in rt_orig.GetAtoms() if a.GetAtomMapNum()}
    )
    assert mapnums

    for m in mapnums:
        a_orig = _find_atom_with_mapnum(rt_orig, m)
        a_copy = _find_atom_with_mapnum(rt_copy, m)
        assert a_orig is not a_copy


@pytest.mark.parametrize(
    ("smiles", "custom_mapping"),
    [
        ("CCO", False),
        ("[CH3:10][CH2:20]O", True),
    ],
)
def test_rdchiralReactants_atoms_r_and_idx_to_mapnum(smiles, custom_mapping):
    rcts = rdchiralReactants(
        smiles, custom_reactant_mapping=custom_mapping, lazy_init=True
    )

    atoms_r = rcts.atoms_r
    assert atoms_r

    if not custom_mapping:
        for idx in range(rcts.reactants.GetNumAtoms()):
            assert rcts.idx_to_mapnum(idx) == idx + 1


def test_rdchiralReactants_reactants_achiral_strips_tetrahedral_and_bond_directionality():
    rcts = rdchiralReactants("F/C=C/F.C[C@H](O)F", custom_reactant_mapping=False)

    achiral = rcts.reactants_achiral

    assert all(
        a.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED
        for a in achiral.GetAtoms()
    )
    assert all(b.GetBondDir() == Chem.rdchem.BondDir.NONE for b in achiral.GetBonds())
    assert all(
        b.GetStereo() == Chem.rdchem.BondStereo.STEREONONE for b in achiral.GetBonds()
    )


def test_rdchiralReaction_reset_restores_original_atom_mapnums():
    """After reset(), all template atoms should have their original map numbers."""
    rxn_smarts = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][Cl:4]"
    r = rdchiralReaction(rxn_smarts, lazy_init=True)

    original_rt_mapnums = {
        a.GetAtomMapNum() for a in r.template_r_orig.GetAtoms() if a.GetAtomMapNum()
    }
    original_pt_mapnums = {
        a.GetAtomMapNum() for a in r.template_p_orig.GetAtoms() if a.GetAtomMapNum()
    }

    # Mutate template atoms' map numbers to simulate post-reaction state
    first_rt_atom = next(iter(r.atoms_rt_map.values()))
    first_rt_atom.SetAtomMapNum(999)
    first_pt_atom = next(iter(r.atoms_pt_map.values()))
    first_pt_atom.SetAtomMapNum(888)

    r.reset()

    reset_rt_mapnums = {
        a.GetAtomMapNum() for a in r.template_r.GetAtoms() if a.GetAtomMapNum()
    }
    reset_pt_mapnums = {
        a.GetAtomMapNum() for a in r.template_p.GetAtoms() if a.GetAtomMapNum()
    }
    assert reset_rt_mapnums == original_rt_mapnums
    assert reset_pt_mapnums == original_pt_mapnums
    assert 999 not in reset_rt_mapnums
    assert 888 not in reset_pt_mapnums


def test_rdchiralReaction_reset_is_in_place_not_copy():
    """reset() should restore mapnums in-place, not create new Mol objects."""
    rxn_smarts = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][Cl:4]"
    r = rdchiralReaction(rxn_smarts, lazy_init=True)

    template_r_before = r.template_r
    template_p_before = r.template_p

    r.reset()

    # Same Mol objects — in-place restore, no copy
    assert r.template_r is template_r_before
    assert r.template_p is template_p_before


def test_rdchiralReaction_reset_keeps_atoms_rt_map_valid():
    """After reset(), atoms_rt_map keys should match the restored map numbers."""
    rxn_smarts = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][Cl:4]"
    r = rdchiralReaction(rxn_smarts, lazy_init=True)

    original_keys = set(r.atoms_rt_map.keys())

    # Mutate a template atom's map number
    first_atom = next(iter(r.atoms_rt_map.values()))
    first_atom.SetAtomMapNum(999)

    r.reset()

    # The dict is not rebuilt, but the atoms' mapnums are restored.
    # The dict keys should still match the original map numbers.
    assert set(r.atoms_rt_map.keys()) == original_keys

    # Each atom in the dict should have its map number matching the key
    for mapnum, atom in r.atoms_rt_map.items():
        assert atom.GetAtomMapNum() == mapnum
