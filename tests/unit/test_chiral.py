import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from rdchiral.chiral import (
    atom_chirality_matches,
    copy_chirality,
    template_atom_could_have_been_tetra,
)


def _atom_from_smarts(smarts: str, mapnum: int) -> Chem.Atom:
    m = Chem.MolFromSmarts(smarts)
    assert m is not None
    for a in m.GetAtoms():
        if a.GetAtomMapNum() == mapnum:
            return a
    raise ValueError(f"No atom with mapnum {mapnum} in smarts {smarts!r}")


def _atom_from_smiles(smiles: str, mapnum: int) -> Chem.Atom:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    for a in m.GetAtoms():
        if a.GetAtomMapNum() == mapnum:
            return a
    raise ValueError(f"No atom with mapnum {mapnum} in smiles {smiles!r}")


def test_template_atom_could_have_been_tetra_degree_three_without_explicit_h_false_and_strips_tag():
    a = _atom_from_smarts("[C:1]([CH3:2])([CH3:3])[CH3:4]", 1)
    a.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

    assert (
        template_atom_could_have_been_tetra(a, strip_if_spec=True, cache=True) is False
    )
    assert a.GetChiralTag() == ChiralType.CHI_UNSPECIFIED
    assert a.HasProp("tetra_possible")
    assert a.GetBoolProp("tetra_possible") is False


def test_template_atom_could_have_been_tetra_degree_three_with_explicit_h_true_and_cached():
    a = _atom_from_smarts("[C:1]([CH3:2])([CH3:3])([CH3:4])[H]", 1)

    assert (
        template_atom_could_have_been_tetra(a, strip_if_spec=False, cache=True) is True
    )
    assert a.HasProp("tetra_possible")
    assert a.GetBoolProp("tetra_possible") is True


def test_template_atom_could_have_been_tetra_respects_existing_cache_value():
    a = _atom_from_smarts("[C:1]([CH3:2])([CH3:3])[H]", 1)
    a.SetBoolProp("tetra_possible", False)

    assert (
        template_atom_could_have_been_tetra(a, strip_if_spec=False, cache=True) is False
    )


def test_atom_chirality_matches_both_unspecified_returns_2():
    a_tmp = _atom_from_smarts("[C:1]", 1)
    a_mol = _atom_from_smiles("[CH3:1]", 1)

    assert atom_chirality_matches(a_tmp, a_mol) == 2


def test_atom_chirality_matches_template_chiral_molecule_unspecified_without_chirality_possible_returns_2():
    a_tmp = _atom_from_smarts("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_mol = _atom_from_smiles("[C:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)

    assert a_mol.GetChiralTag() == ChiralType.CHI_UNSPECIFIED
    assert atom_chirality_matches(a_tmp, a_mol) == 2


def test_atom_chirality_matches_template_chiral_molecule_unspecified_with_chirality_possible_returns_0():
    a_tmp = _atom_from_smarts("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_mol = _atom_from_smiles("[C:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_mol.SetBoolProp("_ChiralityPossible", True)

    assert atom_chirality_matches(a_tmp, a_mol) == 0


def test_atom_chirality_matches_template_unspecified_molecule_chiral_returns_0_when_template_could_have_been_tetra():
    a_tmp = _atom_from_smarts("[C:1]([CH3:2])([CH3:3])[H]", 1)
    a_mol = _atom_from_smiles("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)

    assert a_tmp.GetChiralTag() == ChiralType.CHI_UNSPECIFIED
    assert a_mol.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    assert atom_chirality_matches(a_tmp, a_mol) == 0


def test_atom_chirality_matches_returns_2_when_neighbor_count_ambiguous():
    a_tmp = _atom_from_smarts("[C@:1]([F:2])[Cl:3]", 1)
    a_mol = _atom_from_smiles("[C@:1]([F:2])[Cl:3]", 1)

    assert atom_chirality_matches(a_tmp, a_mol) == 2


@pytest.mark.parametrize(
    ("template_smarts", "mol_smiles", "expected"),
    [
        ("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1),
        (
            "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]",
            "[C@@:1]([F:2])([Cl:3])([Br:4])[I:5]",
            -1,
        ),
    ],
)
def test_atom_chirality_matches_parity_and_tag_controlled_cases(
    template_smarts, mol_smiles, expected
):
    a_tmp = _atom_from_smarts(template_smarts, 1)
    a_mol = _atom_from_smiles(mol_smiles, 1)

    assert atom_chirality_matches(a_tmp, a_mol) == expected


def test_copy_chirality_noop_when_target_degree_less_than_three():
    a_src = _atom_from_smiles("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_new = _atom_from_smiles("[CH2:1][CH3:2]", 1)

    assert a_new.GetDegree() < 3
    copy_chirality(a_src, a_new)
    assert a_new.GetChiralTag() == ChiralType.CHI_UNSPECIFIED


def test_copy_chirality_noop_when_target_degree_three_with_nonsingle_bond():
    a_src = _atom_from_smiles("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_new = _atom_from_smiles("[C:1](=[O:2])([CH3:3])[CH3:4]", 1)

    assert a_new.GetDegree() == 3
    assert any(b.GetBondTypeAsDouble() != 1.0 for b in a_new.GetBonds())
    copy_chirality(a_src, a_new)
    assert a_new.GetChiralTag() == ChiralType.CHI_UNSPECIFIED


def test_copy_chirality_inverts_when_atom_chirality_matches_returns_minus_one():
    a_src = _atom_from_smiles("[C@:1]([F:2])([Cl:3])([Br:4])[I:5]", 1)
    a_new = _atom_from_smiles("[C@:1]([I:2])([F:3])([Cl:4])[Br:5]", 1)

    assert atom_chirality_matches(a_src, a_new) == -1

    copy_chirality(a_src, a_new)

    assert a_src.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW
    assert a_new.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW
