from typing import List, Tuple, cast

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import BondType, ChiralType

from rdchiral.utils import parity4

# Atomic numbers of elements that can be tetrahedral centers at degree 3
# because a lone pair acts as the fourth substituent (e.g., sulfoxides, phosphines).
_LONE_PAIR_STEREO_ATOMS: frozenset[int] = frozenset({15, 16, 33, 34, 51, 52, 83})


def template_atom_could_have_been_tetra(
    a: Chem.Atom, strip_if_spec: bool = False, cache: bool = True
) -> bool:
    """
    Could this atom have been a tetrahedral center?

    If yes, the template atom is considered achiral and will not match a chiral
    reactant. If no, the template atom is auxiliary and should not be used to
    remove a matched reaction (e.g., a fully-generalized terminal [C:1]).

    Atoms such as P, S, As, Se, Sb, Te, and Bi are treated as possible tetrahedral
    centers even at degree 3 because a lone pair can serve as the fourth
    substituent (e.g., phosphines, sulfoxides, selenoxides).

    Args:
        a (rdkit.Chem.rdchem.Atom): RDKit atom.
        strip_if_spec (bool, optional): If True, clear an improperly set chiral
            tag when the atom cannot be tetrahedral. Defaults to False.
        cache (bool, optional): If True, cache the result on the atom via the
            ``tetra_possible`` property. Defaults to True.

    Returns:
        bool: True if the atom could have been a tetrahedral center.
    """

    if a.HasProp("tetra_possible"):
        return bool(a.GetBoolProp("tetra_possible"))
    if a.GetDegree() < 3 or (
        a.GetDegree() == 3
        and "H" not in a.GetSmarts()
        and a.GetAtomicNum() not in _LONE_PAIR_STEREO_ATOMS
    ):
        if cache:
            a.SetBoolProp("tetra_possible", False)
        if strip_if_spec:  # Clear chiral tag in case improperly set
            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
        return False
    if cache:
        a.SetBoolProp("tetra_possible", True)
    return True


def copy_chirality(a_src: Chem.Atom, a_new: Chem.Atom) -> None:
    """
    Copy tetrahedral chirality from one atom to another, inverting if required.

    Atoms such as P, S, As, Se, Sb, Te, and Bi are allowed to carry chirality
    even at degree 3 because a lone pair can act as the fourth substituent.

    Args:
        a_src (rdkit.Chem.rdchem.Atom): Source RDKit atom whose chiral tag will be copied.
        a_new (rdkit.Chem.rdchem.Atom): Destination RDKit atom to receive the chiral tag.

    Returns:
        None: This function mutates `a_new` in place.
    """
    # Not possible to be a tetrahedral center anymore?
    if a_new.GetDegree() < 3:
        return
    if (
        a_new.GetDegree() == 3
        and any(b.GetBondType() != BondType.SINGLE for b in a_new.GetBonds())
        and a_new.GetAtomicNum() not in _LONE_PAIR_STEREO_ATOMS
    ):
        return

    a_new.SetChiralTag(a_src.GetChiralTag())

    if atom_chirality_matches(a_src, a_new) == -1:
        a_new.InvertChirality()


def atom_chirality_matches(a_tmp: Chem.Atom, a_mol: Chem.Atom) -> int:
    """
    Checks for consistency in chirality between a template atom and a molecule atom.

    Also checks to see if chirality needs to be inverted in copy_chirality

    Args:
        a_tmp (rdkit.Chem.rdchem.Atom): Template atom to compare.
        a_mol (rdkit.Chem.rdchem.Atom): Molecule atom to compare against the template.

    Returns:
        int: Integer value of match result:
            +1 if chirality matches and no inversion is needed.
            -1 if chirality matches but is the opposite (inversion needed).
            0 if an explicit NOT match (chiral template vs. achiral molecule that
              could have been chiral).
            2 if the comparison is ambiguous or both atoms are achiral.
    """
    if a_mol.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
        if a_tmp.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
            return 2  # achiral template, achiral molecule -> match
        # What if the template was chiral, but the reactant isn't just due to symmetry?
        if not a_mol.HasProp("_ChiralityPossible"):
            # It's okay to make a match, as long as the product is achiral (even
            # though the product template will try to impose chirality)
            return 2

        # Discussion: figure out if we want this behavior - should a chiral template
        # be applied to an achiral molecule? For the retro case, if we have
        # a retro reaction that requires a specific stereochem, return False;
        # however, there will be many cases where the reaction would probably work
        return 0
    if a_tmp.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
        if template_atom_could_have_been_tetra(a_tmp):
            return 0
        return 2

    mapnums_tmp: List[int] = []

    a_tmp_neighbors = cast(Tuple[rdchem.QueryAtom, ...], a_tmp.GetNeighbors())
    for n in a_tmp_neighbors:
        mapnums_tmp.append(n.GetAtomMapNum())

    mapnums_mol: List[int] = []
    a_mol_neighbors = cast(Tuple[rdchem.QueryAtom, ...], a_mol.GetNeighbors())
    for n in a_mol_neighbors:
        mapnums_mol.append(n.GetAtomMapNum())

    # When there are fewer than 3 heavy neighbors, chirality is ambiguous...
    if len(mapnums_tmp) < 3 or len(mapnums_mol) < 3:
        return 2

    # Degree of 3 -> remaining atom is a hydrogen, add to list
    if len(mapnums_tmp) < 4:
        mapnums_tmp.append(-1)  # H
    if len(mapnums_mol) < 4:
        mapnums_mol.append(-1)  # H

    try:
        only_in_src: List[int] = [i for i in mapnums_tmp if i not in mapnums_mol][
            ::-1
        ]  # reverse for popping
        only_in_mol: List[int] = [i for i in mapnums_mol if i not in mapnums_tmp]
        if len(only_in_src) <= 1 and len(only_in_mol) <= 1:
            tmp_parity = parity4(mapnums_tmp)
            mol_parity = parity4(
                [i if i in mapnums_tmp else only_in_src.pop() for i in mapnums_mol]
            )
            parity_matches = tmp_parity == mol_parity
            tag_matches = a_tmp.GetChiralTag() == a_mol.GetChiralTag()
            chirality_matches = parity_matches == tag_matches
            return 1 if chirality_matches else -1
        else:
            return 2  # ambiguous case, just return for now

    except IndexError as e:
        raise KeyError(f"Pop from empty set - this should not happen! {e}")
