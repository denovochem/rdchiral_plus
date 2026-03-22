import itertools

import pytest
from rdkit import Chem

from rdchiral.utils import atoms_are_different, bond_to_label, parity4


def _parity_expected(values):
    inv = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inv += 1
    return inv % 2


@pytest.mark.parametrize("perm", list(itertools.permutations([1, 2, 3, 4])))
def test_parity4_matches_inversion_parity(perm):
    assert parity4(list(perm)) == _parity_expected(list(perm))


def test_bond_to_label_is_canonical_wrt_endpoint_order():
    m = Chem.MolFromSmiles("[CH3:12][CH2:3]")
    assert m is not None

    bond = m.GetBondBetweenAtoms(0, 1)
    assert bond is not None

    label_forward = bond_to_label(bond)

    bond_reversed = m.GetBondBetweenAtoms(1, 0)
    assert bond_reversed is not None
    label_reverse = bond_to_label(bond_reversed)

    assert label_forward == label_reverse


def test_bond_to_label_includes_bond_smarts_and_atom_map_numbers_when_present():
    m = Chem.MolFromSmiles("[CH3:12]=[CH2:3]")
    assert m is not None

    bond = m.GetBondBetweenAtoms(0, 1)
    assert bond is not None

    label = bond_to_label(bond)

    assert "=" in label
    assert "612" in label
    assert "63" in label


def test_atoms_are_different_identical_atoms_in_identical_molecules_false():
    m1 = Chem.MolFromSmiles("CCO")
    m2 = Chem.MolFromSmiles("CCO")
    assert m1 is not None and m2 is not None

    assert atoms_are_different(m1.GetAtomWithIdx(1), m2.GetAtomWithIdx(1)) is False


def test_atoms_are_different_detects_atomic_number_difference():
    m1 = Chem.MolFromSmiles("C")
    m2 = Chem.MolFromSmiles("O")
    assert m1 is not None and m2 is not None

    assert atoms_are_different(m1.GetAtomWithIdx(0), m2.GetAtomWithIdx(0)) is True


def test_atoms_are_different_detects_degree_or_hydrogen_count_difference():
    m = Chem.MolFromSmiles("CCC")
    assert m is not None

    terminal = m.GetAtomWithIdx(0)
    middle = m.GetAtomWithIdx(1)

    assert atoms_are_different(terminal, middle) is True


def test_atoms_are_different_detects_neighbor_identity_via_bond_labels():
    ethane = Chem.MolFromSmiles("CC")
    methyl_chloride = Chem.MolFromSmiles("CCl")
    assert ethane is not None and methyl_chloride is not None

    c_ethane = ethane.GetAtomWithIdx(0)
    c_methyl_chloride = methyl_chloride.GetAtomWithIdx(0)

    assert c_ethane.GetAtomicNum() == c_methyl_chloride.GetAtomicNum() == 6
    assert c_ethane.GetTotalNumHs() == c_methyl_chloride.GetTotalNumHs() == 3
    assert c_ethane.GetDegree() == c_methyl_chloride.GetDegree() == 1

    assert atoms_are_different(c_ethane, c_methyl_chloride) is True
