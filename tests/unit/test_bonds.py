import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondDir

from rdchiral.bonds import BondDirOpposite, bond_dirs_by_mapnum, enumerate_possible_cistrans_defs


def test_bond_dirs_by_mapnum_adds_opposites_for_mapped_mol() -> None:
    mol = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    assert mol is not None

    dirs = bond_dirs_by_mapnum(mol)

    assert (1, 2) in dirs
    assert (2, 1) in dirs
    assert dirs[(2, 1)] == BondDirOpposite[dirs[(1, 2)]]

    assert (4, 3) in dirs
    assert (3, 4) in dirs
    assert dirs[(3, 4)] == BondDirOpposite[dirs[(4, 3)]]


def test_bond_dirs_by_mapnum_ignores_unmapped_atoms() -> None:
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert mol is not None

    dirs = bond_dirs_by_mapnum(mol)
    assert dirs == {}


def test_enumerate_possible_cistrans_defs_specified_bond_has_16_possible_defs() -> None:
    template_r = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    assert template_r is not None

    required_bond_defs, coreatoms = enumerate_possible_cistrans_defs(template_r)

    assert (2, 3) in coreatoms
    assert (3, 2) in coreatoms

    assert len(required_bond_defs) == 16
    assert all(d1 != BondDir.NONE and d2 != BondDir.NONE for (d1, d2) in required_bond_defs.values())


def test_enumerate_possible_cistrans_defs_unspecified_bond_requires_none_dirs() -> None:
    template_r = Chem.MolFromSmiles("[CH3:1][CH:2]=[CH:3][CH3:4]")
    assert template_r is not None

    required_bond_defs, coreatoms = enumerate_possible_cistrans_defs(template_r)

    assert (2, 3) in coreatoms
    assert (3, 2) in coreatoms

    assert len(required_bond_defs) == 8
    assert set(required_bond_defs.values()) == {(BondDir.NONE, BondDir.NONE)}
