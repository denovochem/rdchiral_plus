import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from rdchiral.template_extractor import (
    clear_mapnum,
    convert_atom_to_wildcard,
    expand_changed_atom_tags,
    extract_from_reaction,
    extract_from_reaction_smiles,
    get_special_groups,
    get_stereogenic_double_bonds,
    get_strict_smarts_for_atom,
    get_tagged_atoms_from_mol,
    get_tagged_atoms_from_mols,
    get_tetrahedral_atoms,
    invert_chirality_around_unmapped_ring_closure,
    mols_from_smiles_list,
    reassign_atom_mapping,
    replace_deuterated,
    split_reaction_smarts,
)

# ---------------------------------------------------------------------------
# replace_deuterated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smi, expected",
    [
        ("[2H]O[2H]", "[H]O[H]"),
        ("CC", "CC"),
        ("[2H]C([2H])([2H])[2H]", "[H]C([H])([H])[H]"),
        ("", ""),
    ],
)
def test_replace_deuterated(smi, expected):
    assert replace_deuterated(smi) == expected


# ---------------------------------------------------------------------------
# clear_mapnum
# ---------------------------------------------------------------------------


def test_clear_mapnum_sets_all_to_zero():
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    assert mol is not None
    result = clear_mapnum(mol)
    assert all(a.GetAtomMapNum() == 0 for a in result.GetAtoms())


def test_clear_mapnum_returns_same_mol_object():
    mol = Chem.MolFromSmiles("[CH3:1]")
    assert mol is not None
    result = clear_mapnum(mol)
    assert result is mol


# ---------------------------------------------------------------------------
# mols_from_smiles_list
# ---------------------------------------------------------------------------


def test_mols_from_smiles_list_valid_smiles():
    mols = mols_from_smiles_list(["CC", "O"])
    assert len(mols) == 2
    assert all(m is not None for m in mols)


def test_mols_from_smiles_list_skips_empty_strings():
    mols = mols_from_smiles_list(["CC", "", "O"])
    assert len(mols) == 2


def test_mols_from_smiles_list_invalid_smiles_gives_none():
    mols = mols_from_smiles_list(["INVALID_XYZ"])
    assert len(mols) == 1
    assert mols[0] is None


def test_mols_from_smiles_list_empty_input():
    assert mols_from_smiles_list([]) == []


# ---------------------------------------------------------------------------
# get_tagged_atoms_from_mol
# ---------------------------------------------------------------------------


def test_get_tagged_atoms_from_mol_returns_mapped_atoms():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2]O")
    assert mol is not None
    atoms, tags = get_tagged_atoms_from_mol(mol)
    assert set(tags) == {1, 2}
    assert len(atoms) == 2


def test_get_tagged_atoms_from_mol_no_mapped_atoms():
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    atoms, tags = get_tagged_atoms_from_mol(mol)
    assert atoms == []
    assert tags == []


# ---------------------------------------------------------------------------
# get_tagged_atoms_from_mols
# ---------------------------------------------------------------------------


def test_get_tagged_atoms_from_mols_aggregates_across_molecules():
    mol1 = Chem.MolFromSmiles("[CH3:1]O")
    mol2 = Chem.MolFromSmiles("[NH2:3]C")
    assert mol1 is not None and mol2 is not None
    atoms, tags = get_tagged_atoms_from_mols([mol1, mol2])
    assert set(tags) == {1, 3}
    assert len(atoms) == 2


# ---------------------------------------------------------------------------
# get_tetrahedral_atoms
# ---------------------------------------------------------------------------


def test_get_tetrahedral_atoms_finds_matching_chiral_center():
    r = Chem.MolFromSmiles("[C@@H:1](F)(Cl)Br")
    p = Chem.MolFromSmiles("[C@H:1](F)(Cl)Br")
    assert r is not None and p is not None
    result = get_tetrahedral_atoms([r], [p])
    assert len(result) == 1
    atom_tag, ar, ap = result[0]
    assert atom_tag == 1
    assert ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    assert ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED


def test_get_tetrahedral_atoms_no_chiral_centers():
    r = Chem.MolFromSmiles("[CH3:1]O")
    p = Chem.MolFromSmiles("[CH3:1]N")
    assert r is not None and p is not None
    result = get_tetrahedral_atoms([r], [p])
    assert result == []


# ---------------------------------------------------------------------------
# get_stereogenic_double_bonds
# ---------------------------------------------------------------------------


def test_get_stereogenic_double_bonds_detects_stereo_change():
    r = Chem.MolFromSmiles(r"[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    p = Chem.MolFromSmiles(r"[CH3:1]/[CH:2]=[CH:3]\[CH3:4]")
    assert r is not None and p is not None
    bonds = get_stereogenic_double_bonds([r], [p])
    assert len(bonds) > 0


def test_get_stereogenic_double_bonds_no_stereo_no_change():
    r = Chem.MolFromSmiles("[CH3:1][CH:2]=[CH:3][CH3:4]")
    p = Chem.MolFromSmiles("[CH3:1][CH:2]=[CH:3][CH3:4]")
    assert r is not None and p is not None
    bonds = get_stereogenic_double_bonds([r], [p])
    assert bonds == []


# ---------------------------------------------------------------------------
# reassign_atom_mapping
# ---------------------------------------------------------------------------


def test_reassign_atom_mapping_sequential_from_one():
    result = reassign_atom_mapping("[C:5][O:3]>>[C:3][O:5]")
    assert result == "[C:1][O:2]>>[C:2][O:1]"


def test_reassign_atom_mapping_already_sequential():
    rxn = "[C:1][O:2]>>[C:1][O:2]"
    assert reassign_atom_mapping(rxn) == rxn


def test_reassign_atom_mapping_consistency_across_arrow():
    result = reassign_atom_mapping("[C:10]>>[C:10]")
    assert result == "[C:1]>>[C:1]"


# ---------------------------------------------------------------------------
# expand_changed_atom_tags
# ---------------------------------------------------------------------------


def test_expand_changed_atom_tags_returns_new_tags():
    expansion = expand_changed_atom_tags([1, 2], "([C:1][O:3])")
    assert expansion == [3]


def test_expand_changed_atom_tags_skips_already_present():
    expansion = expand_changed_atom_tags([1, 2], "([C:1][O:2])")
    assert expansion == []


def test_expand_changed_atom_tags_empty_fragment():
    expansion = expand_changed_atom_tags([1], "")
    assert expansion == []


# ---------------------------------------------------------------------------
# convert_atom_to_wildcard
# ---------------------------------------------------------------------------


def test_convert_atom_to_wildcard_terminal_atom_includes_Hcount():
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    atom = mol.GetAtomWithIdx(0)
    symbol = convert_atom_to_wildcard(atom)
    assert "D1" in symbol
    assert "H3" in symbol


def test_convert_atom_to_wildcard_nonterminal_carbon_uses_C():
    mol = Chem.MolFromSmiles("CCC")
    assert mol is not None
    atom = mol.GetAtomWithIdx(1)
    symbol = convert_atom_to_wildcard(atom)
    assert symbol.startswith("[C;") or symbol == "[C]"


def test_convert_atom_to_wildcard_aromatic_carbon_uses_lowercase_c():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    atom = mol.GetAtomWithIdx(0)
    symbol = convert_atom_to_wildcard(atom)
    assert "c" in symbol


def test_convert_atom_to_wildcard_non_carbon_terminal_does_not_use_atomic_num():
    mol = Chem.MolFromSmiles("CN")
    assert mol is not None
    n_atom = mol.GetAtomWithIdx(1)
    symbol = convert_atom_to_wildcard(n_atom)
    assert "#7" not in symbol


def test_convert_atom_to_wildcard_non_carbon_non_terminal_uses_atomic_num():
    mol = Chem.MolFromSmiles("CNC")
    assert mol is not None
    n_atom = mol.GetAtomWithIdx(1)
    symbol = convert_atom_to_wildcard(n_atom)
    assert "#7" in symbol


# ---------------------------------------------------------------------------
# get_strict_smarts_for_atom
# ---------------------------------------------------------------------------


def test_get_strict_smarts_for_atom_includes_degree_and_Hcount():
    mol = Chem.MolFromSmiles("[CH3:1]CC")
    assert mol is not None
    atom = mol.GetAtomWithIdx(0)
    symbol = get_strict_smarts_for_atom(atom)
    assert "D1" in symbol
    assert "H3" in symbol


def test_get_strict_smarts_for_atom_no_stereo_when_disabled():
    mol = Chem.MolFromSmiles("[C@@H:1](F)(Cl)Br")
    assert mol is not None
    atom = mol.GetAtomWithIdx(0)
    symbol_with = get_strict_smarts_for_atom(atom, use_stereochemistry=True)
    symbol_without = get_strict_smarts_for_atom(atom, use_stereochemistry=False)
    assert "@" in symbol_with
    assert "@" not in symbol_without


# ---------------------------------------------------------------------------
# invert_chirality_around_unmapped_ring_closure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smarts, expected",
    [
        ("[C@@H]1CCCC1", "[C@H]1CCCC1"),
        ("[C@H]1CCCC1", "[C@@H]1CCCC1"),
        ("[C@@H:1]1CCCC1", "[C@@H:1]1CCCC1"),
    ],
)
def test_invert_chirality_around_unmapped_ring_closure(smarts, expected):
    result = invert_chirality_around_unmapped_ring_closure(smarts)
    assert result == expected


def test_invert_chirality_around_unmapped_ring_closure_no_match_unchanged():
    smarts = "[C:1](F)(Cl)Br"
    assert invert_chirality_around_unmapped_ring_closure(smarts) == smarts


# ---------------------------------------------------------------------------
# get_special_groups
# ---------------------------------------------------------------------------


def test_get_special_groups_finds_boronic_acid():
    mol = Chem.MolFromSmiles("OB(O)c1ccccc1")
    assert mol is not None
    groups = get_special_groups(mol)
    assert len(groups) > 0


def test_get_special_groups_no_groups_in_simple_alkane():
    mol = Chem.MolFromSmiles("CCCC")
    assert mol is not None
    groups = get_special_groups(mol)
    assert groups == []


# ---------------------------------------------------------------------------
# split_reaction_smarts
# ---------------------------------------------------------------------------


def test_split_reaction_smarts_single_component():
    result = split_reaction_smarts("[C:1]>>[C:1]")
    assert len(result) == 1
    assert result[0] == "[C:1]>>[C:1]"


def test_split_reaction_smarts_two_independent_components():
    result = split_reaction_smarts("[C:1].[N:2]>>[C:1].[N:2]")
    assert len(result) == 2
    smarts_set = set(result)
    assert "[C:1]>>[C:1]" in smarts_set
    assert "[N:2]>>[N:2]" in smarts_set


# ---------------------------------------------------------------------------
# extract_from_reaction
# ---------------------------------------------------------------------------


def test_extract_from_reaction_simple_returns_nonempty_template():
    reaction = {
        "reactants": "[CH3:1][OH:2].[Cl:3][C:4](=O)[CH3:5]",
        "products": "[CH3:1][O:2][C:4](=O)[CH3:5]",
        "_id": "test_1",
    }
    result = extract_from_reaction(reaction)
    assert result["reaction_smarts"] != ""
    assert result["reaction_id"] == "test_1"


def test_extract_from_reaction_invalid_reactants_returns_default():
    reaction = {"reactants": "INVALID", "products": "[CH3:1]O", "_id": None}
    result = extract_from_reaction(reaction)
    assert result["reaction_smarts"] == ""


def test_extract_from_reaction_no_changed_atoms_returns_default():
    reaction = {
        "reactants": "[CH3:1][OH:2]",
        "products": "[CH3:1][OH:2]",
        "_id": None,
    }
    result = extract_from_reaction(reaction)
    assert result["reaction_smarts"] == ""


def test_extract_from_reaction_preserves_reaction_id():
    reaction = {
        "reactants": "INVALID",
        "products": "INVALID",
        "_id": 42,
    }
    result = extract_from_reaction(reaction)
    assert result["reaction_id"] == 42


def test_extract_from_reaction_too_many_unmapped_product_atoms_returns_default():
    reaction = {
        "reactants": "[CH3:1]O",
        "products": "[CH3:1]OCCCCC",
        "_id": None,
    }
    result = extract_from_reaction(reaction, maximum_number_unmapped_product_atoms=1)
    assert result["reaction_smarts"] == ""


# ---------------------------------------------------------------------------
# extract_from_reaction_smiles
# ---------------------------------------------------------------------------


def test_extract_from_reaction_smiles_wrapper_returns_same_as_dict():
    rxn_smiles = "[CH3:1][OH:2].[Cl:3][C:4](=O)[CH3:5]>>[CH3:1][O:2][C:4](=O)[CH3:5]"
    result = extract_from_reaction_smiles(rxn_smiles)
    assert result["reaction_smarts"] != ""


def test_extract_from_reaction_smiles_raises_on_bad_format():
    with pytest.raises(ValueError):
        extract_from_reaction_smiles("CCO")


def test_extract_from_reaction_smiles_passes_reaction_id():
    rxn_smiles = "[CH3:1][OH:2].[Cl:3][C:4](=O)[CH3:5]>>[CH3:1][O:2][C:4](=O)[CH3:5]"
    result = extract_from_reaction_smiles(rxn_smiles, reaction_id="rxn_99")
    assert result["reaction_id"] == "rxn_99"
