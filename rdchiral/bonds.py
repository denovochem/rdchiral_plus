from typing import Dict, List, Optional, Set, Tuple

import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondDir, BondType

BondDirOpposite = {
    BondDir.ENDUPRIGHT: BondDir.ENDDOWNRIGHT,
    BondDir.ENDDOWNRIGHT: BondDir.ENDUPRIGHT,
    BondDir.NONE: BondDir.NONE,
}
BondDirLabel = {BondDir.ENDUPRIGHT: "\\", BondDir.ENDDOWNRIGHT: "/"}


def bond_dirs_by_mapnum(mol: Chem.Mol) -> Dict[Tuple[int, int], BondDir]:
    """
    Return bond directions keyed by atom-map numbers for a mapped RDKit molecule.

    Args:
        mol (Chem.Mol): RDKit molecule whose bonds will be inspected.

    Returns:
        Dict[Tuple[int, int], BondDir]: Mapping from `(mapnum1, mapnum2)` to the
            bond direction for the bond from atom `mapnum1` to atom `mapnum2`.

    Note:
        Bonds with `BondDir.NONE` are ignored.
        Bonds are also ignored if either endpoint atom has atom-map number 0
        (i.e., is unmapped).
        For each directed entry `(a, b) -> d`, an opposite-direction entry
        `(b, a) -> BondDirOpposite[d]` is also added.
    """
    bond_dirs_by_mapnum: Dict[Tuple[int, int], BondDir] = {}
    for b in mol.GetBonds():
        bond_dir = b.GetBondDir()
        if bond_dir == BondDir.NONE:
            continue
        begin_atom_map_num = b.GetBeginAtom().GetAtomMapNum()
        if not begin_atom_map_num:
            continue
        end_atom_map_num = b.GetEndAtom().GetAtomMapNum()
        if not end_atom_map_num:
            continue
        bond_dirs_by_mapnum[(begin_atom_map_num, end_atom_map_num)] = bond_dir
        bond_dirs_by_mapnum[(end_atom_map_num, begin_atom_map_num)] = BondDirOpposite[
            bond_dir
        ]
    return bond_dirs_by_mapnum


def enumerate_possible_cistrans_defs(
    template_r: Chem.Mol,
) -> Tuple[
    Dict[Tuple[int, int, int, int], Tuple[BondDir, BondDir]], Set[Tuple[int, int]]
]:
    """
    This function is meant to take a reactant template and fully enumerate
    all the ways in which different double-bonds can have their cis/trans
    chirality specified. This is necessary because double-bond chirality cannot
    be specified using cis/trans (global properties), but must be done using
    ENDUPRIGHT and ENDDOWNRIGHT for the attached single bonds (local properties).
    Now, the next issue is that on each side of the double bond, only one of
    the single bond directions must be specified, and that direction can be
    using either atom order. e.g.,

    A1         B1
       \      /
         C = C
       /      \
    A2         B2

    Can be specified by:
    A1-C is an ENDDOWNRIGHT, C-B1 is an ENDUPRIGHT
    A1-C is an ENDDOWNRIGHT, C-B2 is an ENDDOWNRIGHT
    A1-C is an ENDDOWNRIGHT, B1-C is an ENDDOWNRIGHT
    A1-C is an ENDDOWNRIGHT, B2-C is an ENDUPRIGHT
    ...and twelve more definitions using different A1/A2 specs.

    ALSO - we can think about horizontally reflecting this bond entirely,
    which gets us even more definitions.

    So, the point of this function is to fully enumerate *all* of the ways
    in which chirality could have been specified. That way, we can take a
    reactant atom and check if its chirality is within the list of acceptable
    definitions to determine if a match was made.

    The way we do this is by first defining the *local* chirality of a double
    bond, which weights side chains based purely on the unique mapnum numbering.
    Once we have a local cis/trans definition for a double bond, we can enumerate
    the sixteen possible ways that a reactant could match it.

    Args:
        template_r: reactant template
    
    Returns:
        (dict, set): Returns required_bond_defs and required_bond_defs_coreatoms
    """

    required_bond_defs: Dict[Tuple[int, int, int, int], Tuple[BondDir, BondDir]] = {}
    required_bond_defs_coreatoms: Set[Tuple[int, int]] = set()

    for b in template_r.GetBonds():
        if b.GetBondType() != BondType.DOUBLE:
            continue

        # Define begin and end atoms of the double bond
        ba = b.GetBeginAtom()
        bb = b.GetEndAtom()

        # Now check if it is even possible to specify
        if ba.GetDegree() == 1 or bb.GetDegree() == 1:
            continue

        ba_label = ba.GetAtomMapNum()
        bb_label = bb.GetAtomMapNum()

        # Save core atoms so we know that cis/trans was POSSIBLE to specify
        required_bond_defs_coreatoms.add((ba_label, bb_label))
        required_bond_defs_coreatoms.add((bb_label, ba_label))

        # Define heaviest mapnum neighbor for each atom, excluding the other side of the double bond
        ba_neighbor_labels: List[int] = [a.GetAtomMapNum() for a in ba.GetNeighbors()]
        ba_neighbor_labels.remove(bb_label)  # remove other side of =
        ba_neighbor_labels_max = max(ba_neighbor_labels)
        bb_neighbor_labels: List[int] = [a.GetAtomMapNum() for a in bb.GetNeighbors()]
        bb_neighbor_labels.remove(ba_label)  # remove other side of =
        bb_neighbor_labels_max = max(bb_neighbor_labels)

        # The direction of the bond being observed might need to be flipped,
        # based on
        #     (a) if it is the heaviest atom on this side, and
        #     (b) if the begin/end atoms for the directional bond are
        #         in the wrong order (i.e., if the double-bonded atom
        #         is the begin atom)
        front_spec = None
        back_spec = None
        for bab in ba.GetBonds():
            if bab.GetBondDir() != BondDir.NONE:
                if bab.GetBeginAtom().GetAtomMapNum() == ba_label:
                    # Bond is in wrong order - flip
                    if bab.GetEndAtom().GetAtomMapNum() != ba_neighbor_labels_max:
                        # Defined atom is not the heaviest - flip
                        front_spec = bab.GetBondDir()
                        break
                    front_spec = BondDirOpposite[bab.GetBondDir()]
                    break
                if bab.GetBeginAtom().GetAtomMapNum() != ba_neighbor_labels_max:
                    # Defined atom is not heaviest
                    front_spec = BondDirOpposite[bab.GetBondDir()]
                    break
                front_spec = bab.GetBondDir()
                break
        if front_spec is not None:
            for bbb in bb.GetBonds():
                if bbb.GetBondDir() != BondDir.NONE:
                    # For the "back" specification, the double-bonded atom *should* be the BeginAtom
                    if bbb.GetEndAtom().GetAtomMapNum() == bb_label:
                        # Bond is in wrong order - flip
                        if bbb.GetBeginAtom().GetAtomMapNum() != bb_neighbor_labels_max:
                            # Defined atom is not the heaviest - flip
                            back_spec = bbb.GetBondDir()
                            break
                        back_spec = BondDirOpposite[bbb.GetBondDir()]
                        break
                    if bbb.GetEndAtom().GetAtomMapNum() != bb_neighbor_labels_max:
                        # Defined atom is not heaviest - flip
                        back_spec = BondDirOpposite[bbb.GetBondDir()]
                        break
                    back_spec = bbb.GetBondDir()
                    break

        # Is this an overall unspecified bond? Put it in the dictionary anyway,
        # so there is something to match
        if front_spec is None or back_spec is None:
            # Create a definition over this bond so that reactant MUST be unspecified, too
            for start_atom in ba_neighbor_labels:
                for end_atom in bb_neighbor_labels:
                    required_bond_defs[(start_atom, ba_label, bb_label, end_atom)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(ba_label, start_atom, bb_label, end_atom)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(start_atom, ba_label, end_atom, bb_label)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(ba_label, start_atom, end_atom, bb_label)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(bb_label, end_atom, start_atom, ba_label)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(end_atom, bb_label, start_atom, ba_label)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(bb_label, end_atom, ba_label, start_atom)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
                    required_bond_defs[(end_atom, bb_label, ba_label, start_atom)] = (
                        BondDir.NONE,
                        BondDir.NONE,
                    )
            continue

        if front_spec == back_spec:
            b.SetProp("localChirality", "trans")
        else:
            b.SetProp("localChirality", "cis")

        possible_defs = {}
        for start_atom in ba_neighbor_labels:
            for end_atom in bb_neighbor_labels:
                needs_inversion = (start_atom != ba_neighbor_labels_max) != (
                    end_atom != bb_neighbor_labels_max
                )
                for start_atom_dir in [BondDir.ENDUPRIGHT, BondDir.ENDDOWNRIGHT]:
                    # When locally trans, BondDir of start shold be same as end,
                    # unless we need inversion
                    if (front_spec != back_spec) != needs_inversion:
                        # locally cis and does not need inversion (True, False)
                        # or locally trans and does need inversion (False, True)
                        end_atom_dir = BondDirOpposite[start_atom_dir]
                    else:
                        # locally cis and does need inversion (True, True)
                        # or locally trans and does not need inversion (False, False)
                        end_atom_dir = start_atom_dir

                    # Enumerate all combinations of atom orders...
                    possible_defs[(start_atom, ba_label, bb_label, end_atom)] = (
                        start_atom_dir,
                        end_atom_dir,
                    )
                    possible_defs[(ba_label, start_atom, bb_label, end_atom)] = (
                        BondDirOpposite[start_atom_dir],
                        end_atom_dir,
                    )
                    possible_defs[(start_atom, ba_label, end_atom, bb_label)] = (
                        start_atom_dir,
                        BondDirOpposite[end_atom_dir],
                    )
                    possible_defs[(ba_label, start_atom, end_atom, bb_label)] = (
                        BondDirOpposite[start_atom_dir],
                        BondDirOpposite[end_atom_dir],
                    )

                    possible_defs[(bb_label, end_atom, start_atom, ba_label)] = (
                        end_atom_dir,
                        start_atom_dir,
                    )
                    possible_defs[(bb_label, end_atom, ba_label, start_atom)] = (
                        end_atom_dir,
                        BondDirOpposite[start_atom_dir],
                    )
                    possible_defs[(end_atom, bb_label, start_atom, ba_label)] = (
                        BondDirOpposite[end_atom_dir],
                        start_atom_dir,
                    )
                    possible_defs[(end_atom, bb_label, ba_label, start_atom)] = (
                        BondDirOpposite[end_atom_dir],
                        BondDirOpposite[start_atom_dir],
                    )

        # Save to the definition of this bond (in either direction)
        required_bond_defs.update(possible_defs)

    return required_bond_defs, required_bond_defs_coreatoms


def get_atoms_across_double_bonds(
    mol: Chem.Mol,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[BondDir, BondDir], bool]]:
    """
    This function takes a molecule and returns a list of cis/trans specifications
    according to the following:

    (mapnums, dirs)

    where atoms = (a1, a2, a3, a4) and dirs = (d1, d2)
    and (a1, a2) defines the ENDUPRIGHT/ENDDOWNRIGHT direction of the "front"
    of the bond using d1, and (a3, a4) defines the direction of the "back" of
    the bond using d2.

    This is used to initialize reactants with a SINGLE definition constraining
    the chirality. Templates have their chirality fully enumerated, so we can
    match this specific definition to the full set of possible definitions
    when determining if a match should be made.

    NOTE: the atom mapnums are returned. This is so we can later use them
    to get the old_mapno property from the corresponding product atom, which is
    an outcome-specific assignment

    We also include implicit chirality here based on ring membership, but keep
    track of that separately

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule

    Returns:
        list: atoms_across_double_bonds
    """
    atoms_across_double_bonds: List[
        Tuple[Tuple[int, int, int, int], Tuple[BondDir, BondDir], bool]
    ] = []
    ring_info: Chem.RingInfo = mol.GetRingInfo()
    atomrings = ring_info.AtomRings()

    for b in mol.GetBonds():
        if b.GetBondType() != BondType.DOUBLE:
            continue

        # Define begin and end atoms of the double bond
        ba = b.GetBeginAtom()
        bb = b.GetEndAtom()

        # Now check if it is even possible to specify
        if ba.GetDegree() == 1 or bb.GetDegree() == 1:
            continue

        ba_label = ba.GetAtomMapNum()
        bb_label = bb.GetAtomMapNum()

        # Try to specify front and back direction separately
        front_mapnums: Optional[Tuple[int, int]] = None
        front_dir: Optional[BondDir] = None
        back_mapnums: Optional[Tuple[int, int]] = None
        back_dir: Optional[BondDir] = None
        is_implicit = False
        bab = None
        bbb = None

        def _bab_generator():
            for z in ba.GetBonds():
                if z.GetBondType() != BondType.DOUBLE:
                    yield z

        for bab in _bab_generator():
            if bab.GetBondDir() != BondDir.NONE:
                front_mapnums = (
                    bab.GetBeginAtom().GetAtomMapNum(),
                    bab.GetEndAtom().GetAtomMapNum(),
                )
                front_dir = bab.GetBondDir()
                break

        def _bbb_generator():
            for z in bb.GetBonds():
                if z.GetBondType() != BondType.DOUBLE:
                    yield z

        for bbb in _bbb_generator():
            if bbb.GetBondDir() != BondDir.NONE:
                back_mapnums = (
                    bbb.GetBeginAtom().GetAtomMapNum(),
                    bbb.GetEndAtom().GetAtomMapNum(),
                )
                back_dir = bbb.GetBondDir()
                break

        # If impossible to spec, just continue
        if bab is None or bbb is None:
            continue

        # Did we actually get a specification out?
        if front_dir is None or back_dir is None:
            if b.IsInRing():
                # Implicit cis! Now to figure out right definitions...
                for atomring in atomrings:
                    if ba.GetIdx() in atomring and bb.GetIdx() in atomring:
                        front_mapnums = (bab.GetOtherAtom(ba).GetAtomMapNum(), ba_label)
                        back_mapnums = (bb_label, bbb.GetOtherAtom(bb).GetAtomMapNum())
                        if (bab.GetOtherAtomIdx(ba.GetIdx()) in atomring) != (
                            bbb.GetOtherAtomIdx(bb.GetIdx()) in atomring
                        ):
                            # one of these atoms are in the ring, one is outside -> trans
                            front_dir = BondDir.ENDUPRIGHT
                            back_dir = BondDir.ENDUPRIGHT
                        else:
                            front_dir = BondDir.ENDUPRIGHT
                            back_dir = BondDir.ENDDOWNRIGHT
                        is_implicit = True
                        break

            else:
                # Okay no, actually unspecified
                # Specify direction as BondDir.NONE using whatever bab and bbb were at the end fo the loop
                # note: this is why we use "for bab in ___generator___", so that we know the current
                #       value of bab and bbb correspond to a single bond we can def. by
                front_mapnums = (
                    bab.GetBeginAtom().GetAtomMapNum(),
                    bab.GetEndAtom().GetAtomMapNum(),
                )
                front_dir = BondDir.NONE
                back_mapnums = (
                    bbb.GetBeginAtom().GetAtomMapNum(),
                    bbb.GetEndAtom().GetAtomMapNum(),
                )
                back_dir = BondDir.NONE

        if front_mapnums is None or back_mapnums is None:
            raise ValueError("Could not find mapnums for double bond")
        if front_dir is None or back_dir is None:
            raise ValueError("Could not find bond directions for double bond")
        # Save this (a1, a2, a3, a4) -> (d1, d2) spec
        atoms_across_double_bonds.append(
            (
                front_mapnums + back_mapnums,
                (front_dir, back_dir),
                is_implicit,
            )
        )

    return atoms_across_double_bonds


def restore_bond_stereo_to_sp2_atom(
    a: Chem.Atom, bond_dirs_by_mapnum: Dict[Tuple[int, int], BondDir]
) -> bool:
    """
    Copy over single-bond directions (ENDUPRIGHT, ENDDOWNRIGHT) to
    the single bonds attached to some double-bonded atom, a

    In some cases, like C=C/O>>C=C/Br, we should assume that stereochem was
    preserved, even though mapnums won't match. There might be some reactions
    where the chirality is inverted (like C=C/O >> C=C\Br), but let's not
    worry about those for now...

    Args:
        a (rdkit.Chem.rdchem.Atom): RDKit atom with double bond
        bond_dirs_by_mapnum - dictionary of (begin_mapnum, end_mapnum): bond_dir
            that defines if a bond should be ENDUPRIGHT or ENDDOWNRIGHT. The reverse
            key is also included with the reverse bond direction. If the source
            molecule did not have a specified chirality at this double bond, then
            the mapnum tuples will be missing from the dict
    Returns:
        bool: Returns Trueif a bond direction was copied
    """

    for bond_to_spec in a.GetBonds():
        if (
            bond_to_spec.GetOtherAtom(a).GetAtomMapNum(),
            a.GetAtomMapNum(),
        ) in bond_dirs_by_mapnum:
            bond_to_spec.SetBondDir(
                bond_dirs_by_mapnum[
                    (
                        bond_to_spec.GetBeginAtom().GetAtomMapNum(),
                        bond_to_spec.GetEndAtom().GetAtomMapNum(),
                    )
                ]
            )
            return True

    if a.GetDegree() == 2:
        # Either the branch used to define was replaced with H (deg 3 -> deg 2)
        # or the branch used to define was reacted (deg 2 -> deg 2)
        for bond_to_spec in a.GetBonds():
            if bond_to_spec.GetBondType() == BondType.DOUBLE:
                continue
            if not bond_to_spec.GetOtherAtom(a).HasProp("old_mapno"):
                # new atom, deg2->deg2, assume direction preserved
                needs_inversion = False
            else:
                # old atom, just was not used in chirality definition - set opposite
                needs_inversion = True

            for (i, j), bond_dir in bond_dirs_by_mapnum.items():
                if bond_dir != BondDir.NONE:
                    if i == bond_to_spec.GetBeginAtom().GetAtomMapNum():
                        if needs_inversion:
                            bond_to_spec.SetBondDir(BondDirOpposite[bond_dir])
                        else:
                            bond_to_spec.SetBondDir(bond_dir)
                        return True

    elif a.GetDegree() == 3:
        # If we lost the branch defining stereochem, it must have been replaced
        for bond_to_spec in a.GetBonds():
            if bond_to_spec.GetBondType() == BondType.DOUBLE:
                continue
            oa = bond_to_spec.GetOtherAtom(a)
            if oa.HasProp("old_mapno") or oa.HasProp("react_atom_idx"):
                # looking at an old atom, which should have opposite direction as removed atom
                needs_inversion = True
            else:
                # looking at a new atom, assume same as removed atom
                needs_inversion = False

            for (i, j), bond_dir in bond_dirs_by_mapnum.items():
                if bond_dir != BondDir.NONE:
                    if i == bond_to_spec.GetBeginAtom().GetAtomMapNum():
                        if needs_inversion:
                            bond_to_spec.SetBondDir(BondDirOpposite[bond_dir])
                        else:
                            bond_to_spec.SetBondDir(bond_dir)
                        return True

    return False


def correct_conjugated(
    initial_bond_dirs: Dict[Tuple[int, int], BondDir], outcome: Chem.Mol
) -> bool:
    """
    Checks whether the copying over of single-bond directions (ENDUPRIGHT, ENDDOWNRIGHT) was
    corrupted for a conjugated system, where parts of the directions were specified by the template
    and parts were copied from the reactants.
    Args:
        initial_bond_dirs - dictionary of (begin_mapnum, end_mapnum): bond_dir
            that defines if a bond is ENDUPRIGHT or ENDDOWNRIGHT. The reverse
            key is also included with the reverse bond direction. If the source
            molecule did not have a specified chirality at this double bond, then
            the mapnum tuples will be missing from the dict
        outcome (rdkit.Chem.rdChem.Mol): RDKit molecule
    Returns:
        bool: Returns True if a conjugated system was corrected
    """

    if not initial_bond_dirs:
        return False

    conjugated: List[Tuple[int, int]] = []
    for b in outcome.GetBonds():
        if b.GetIsConjugated():
            conjugated.append(
                (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())
            )

    if not conjugated:
        return False

    final_bond_dirs = bond_dirs_by_mapnum(outcome)

    # connectivity of conjugated systems
    # DFS may be better for large systems
    isolated_conjugated: List[Set[int]] = []
    for i, j in conjugated:
        found = False
        for c in isolated_conjugated:
            if i in c or j in c:
                found = True
                c.add(i)
                c.add(j)
                break
        if not found:
            isolated_conjugated.append({i, j})

    need_to_change_dirs = {}
    inverted_dirs = []
    new_dirs = []
    for pair in final_bond_dirs:
        if pair in initial_bond_dirs:
            if final_bond_dirs[pair] != initial_bond_dirs[pair]:
                need_to_change_dirs[pair] = initial_bond_dirs[pair]
                inverted_dirs.append(pair)
        else:
            new_dirs.append(pair)

    if not inverted_dirs:
        return False

    isolated_conjugated_need_fix = [False] * len(isolated_conjugated)
    for i, conj in enumerate(isolated_conjugated):
        for pair in inverted_dirs:
            if pair[0] in conj or pair[1] in conj:
                isolated_conjugated_need_fix[i] = True
                break

    for pair in new_dirs:
        for need_fix, conj in zip(isolated_conjugated_need_fix, isolated_conjugated):
            if need_fix and (pair[0] in conj or pair[1] in conj):
                need_to_change_dirs[pair] = BondDirOpposite[final_bond_dirs[pair]]
                break

    changed = False
    for b in outcome.GetBonds():
        bam = b.GetBeginAtom().GetAtomMapNum()
        bbm = b.GetEndAtom().GetAtomMapNum()
        if (bam, bbm) in need_to_change_dirs:
            b.SetBondDir(need_to_change_dirs[(bam, bbm)])
            changed = True

    return changed
