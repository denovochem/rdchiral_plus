from __future__ import print_function

from typing import List, Tuple

from rdkit import Chem


def parity4(data: List[int]) -> int:
    """
    Compute the parity (even/odd) of the permutation implied by ordering four values.

    Args:
        data (List[int]): A list of four integer values whose relative ordering defines a permutation.

    Returns:
        int: `0` for even parity and `1` for odd parity.

    Note:
        This is an unrolled, branch-based parity computation specialized to length-4 inputs.
        It assumes `data` has length 4 and is intended for inputs with distinct values; ties
        can lead to parity results that are not meaningful.
        Based on: http://www.dalkescientific.com/writings/diary/archive/2016/08/15/fragment_parity_calculation.html
    """
    if data[0] < data[1]:
        if data[2] < data[3]:
            if data[0] < data[2]:
                if data[1] < data[2]:
                    return 0  # (0, 1, 2, 3)
                else:
                    if data[1] < data[3]:
                        return 1  # (0, 2, 1, 3)
                    else:
                        return 0  # (0, 3, 1, 2)
            else:
                if data[0] < data[3]:
                    if data[1] < data[3]:
                        return 0  # (1, 2, 0, 3)
                    else:
                        return 1  # (1, 3, 0, 2)
                else:
                    return 0  # (2, 3, 0, 1)
        else:
            if data[0] < data[3]:
                if data[1] < data[2]:
                    if data[1] < data[3]:
                        return 1  # (0, 1, 3, 2)
                    else:
                        return 0  # (0, 2, 3, 1)
                else:
                    return 1  # (0, 3, 2, 1)
            else:
                if data[0] < data[2]:
                    if data[1] < data[2]:
                        return 1  # (1, 2, 3, 0)
                    else:
                        return 0  # (1, 3, 2, 0)
                else:
                    return 1  # (2, 3, 1, 0)
    else:
        if data[2] < data[3]:
            if data[0] < data[3]:
                if data[0] < data[2]:
                    return 1  # (1, 0, 2, 3)
                else:
                    if data[1] < data[2]:
                        return 0  # (2, 0, 1, 3)
                    else:
                        return 1  # (2, 1, 0, 3)
            else:
                if data[1] < data[2]:
                    return 1  # (3, 0, 1, 2)
                else:
                    if data[1] < data[3]:
                        return 0  # (3, 1, 0, 2)
                    else:
                        return 1  # (3, 2, 0, 1)
        else:
            if data[0] < data[2]:
                if data[0] < data[3]:
                    return 0  # (1, 0, 3, 2)
                else:
                    if data[1] < data[3]:
                        return 1  # (2, 0, 3, 1)
                    else:
                        return 0  # (2, 1, 3, 0)
            else:
                if data[1] < data[2]:
                    if data[1] < data[3]:
                        return 0  # (3, 0, 2, 1)
                    else:
                        return 1  # (3, 1, 2, 0)
                else:
                    return 0  # (3, 2, 1, 0)


def bond_to_label(bond: Chem.Bond) -> str:
    """
    Create a canonical string label for an RDKit bond based on its endpoint atoms and bond SMARTS.

    Args:
        bond (Chem.Bond): RDKit bond object.

    Returns:
        str: Canonical label encoding the two endpoint atoms (atomic number plus optional atom-map number)
            and the bond SMARTS.
    """

    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().GetAtomMapNum():
        a1_label += str(bond.GetBeginAtom().GetAtomMapNum())
    if bond.GetEndAtom().GetAtomMapNum():
        a2_label += str(bond.GetEndAtom().GetAtomMapNum())
    atoms = sorted([a1_label, a2_label])

    return "{}{}{}".format(atoms[0], Chem.Bond.GetSmarts(bond), atoms[1])


def atoms_are_different(atom1: Chem.Atom, atom2: Chem.Atom) -> bool:
    """Return True if two RDKit atoms differ by selected atomic or local-environment properties.

    Args:
        atom1 (Chem.Atom): First atom to compare.
        atom2 (Chem.Atom): Second atom to compare.

    Returns:
        bool: True if any checked property differs, otherwise False.
    """

    if atom1.GetSmarts() != atom2.GetSmarts():
        return True  # should be very general
    if atom1.GetAtomicNum() != atom2.GetAtomicNum():
        return True  # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs():
        return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        return True
    if atom1.GetDegree() != atom2.GetDegree():
        return True
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic():
        return True

    # Check bonds and nearest neighbor identity
    atom_1_bonds: Tuple[Chem.Bond, ...] = tuple(atom1.GetBonds())
    bonds1: List[str] = []
    for bond in atom_1_bonds:
        bonds1.append(bond_to_label(bond))
    bonds1 = sorted(bonds1)

    atom_2_bonds: Tuple[Chem.Bond, ...] = tuple(atom2.GetBonds())
    bonds2: List[str] = []
    for bond in atom_2_bonds:
        bonds2.append(bond_to_label(bond))
    bonds2 = sorted(bonds2)

    if bonds1 != bonds2:
        return True

    return False
