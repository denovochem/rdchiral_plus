from __future__ import print_function

from typing import Any, Dict, List, Tuple

import rdkit.Chem as Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondDir, BondType, ChiralType

from rdchiral.bonds import (
    BondDirOpposite,
    bond_dirs_by_mapnum,
    correct_conjugated,
    restore_bond_stereo_to_sp2_atom,
)
from rdchiral.chiral import (
    atom_chirality_matches,
    copy_chirality,
    template_atom_could_have_been_tetra,
)
from rdchiral.clean import combine_enantiomers_into_racemic
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.utils import atoms_are_different

"""
This file contains the main functions for running reactions. 

An incomplete description of expected behavior is as follows:

(1) RDKit's native RunReactants is called on an achiral version of the molecule,
which has had all tetrahedral centers and bond directions stripped.

(2) For each outcome, we examine the correspondence between atoms in the
reactants and atoms in the reactant template for reasons to exclude the 
current outcome. The way we do so is through the react_atom_idx property in
the generated products. This is one of the 
few properties always copied over to the products here:
https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionRunner.cpp

A previous version of this code did so through the Isotope label of each atom,
before the react_atom_idx was added to the ReactionRunner.cpp code.

The following conditions are checked:

    TETRAHEDRAL ATOMS
    (a) If a reactant atom is a tetrahedral center with specified chirality
        and the reactant template atom is NOT chiral but is defined in a way
        that it could have been specified, reject this outcome
    (b) If a reactant atom is a tetrahedral center with specified chirality
        and the reactant template atom is NOT chiral and is not defined in
        a way where it could have been (i.e., is generalized without spec.
        neighbors), then keep the match.
    (c) If a reactant atom is achiral but the reactant tempalte atom is chiral,
        the match is still allowed to happen. We might want to change this later
        or let it be an option.
    (d) If a reactant atom is a tetrahedral center with specified chirality
        and the reactant template also has its chirality specified, let the
        match happen if the chirality matches.


    DOUBLE BONDS
    (a) If a reactant double bond is defined with directionality specified and
        the reactant template is unspecified but COULD have been (i.e., 
        neighbors of sp2 carbons are specified), reject this outcome
    (b) If a reactant double bond is defined with directionality specified and
        the reactant template si unspecified but could NOT have been (in the
        case of generalization), allow the match to occur. This is what we
        default to when half the bond is specified, like in "C=C/O"
    note: reactants are checked for implicit bond stereo based on rings
    (c) If a reactant double bond has implicit cis due to ring membership, it is
        still allowed to match an unspecified template double bond. Might lead
        to some weird edge cases, but mostly makes sense.


(3) For each outcome, merge all products into a single molecule. During this
process, we check for bonds that are missing in the product. These are those
that were present in the reactants but were NOT matched in the reactant
template.

(4) For each outcome, examine product atoms to correct tetrahedral chirality.

(5) For each outcome, examine product double bonds to correct cis/trans-ness

"""


def rdchiralRunText(
    reaction_smarts: str,
    reactant_smiles: str,
    custom_reactant_mapping: bool = False,
    keep_mapnums: bool = False,
    combine_enantiomers: bool = True,
    return_mapped: bool = False,
) -> List[str] | Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]:
    """Run from SMARTS string and SMILES string. This is NOT recommended
    for library application, since initialization is pretty slow. You should
    separately initialize the template and molecules and call run()

    Args:
        reaction_smarts (str): Reaction SMARTS string
        reactant_smiles (str): Reactant SMILES string
        **kwargs: passed through to `rdchiralRun`

    Returns:
        list: List of outcomes from `rdchiralRun`
    """
    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles, custom_reactant_mapping)
    return rdchiralRun(rxn, reactants, keep_mapnums, combine_enantiomers, return_mapped)


def rdchiralRun(
    rxn: rdchiralReaction,
    reactants: rdchiralReactants,
    keep_mapnums: bool = False,
    combine_enantiomers: bool = True,
    return_mapped: bool = False,
) -> List[str] | Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]:
    """Run rdchiral reaction

    NOTE: there is a fair amount of initialization (assigning stereochem), most
    importantly assigning atom map numbers to the reactant atoms. It is
    HIGHLY recommended to use the custom classes for initialization.

    Args:
        rxn (rdchiralReaction): (rdkit reaction + auxilliary information)
        reactants (rdchiralReactants): (rdkit mol + auxilliary information)
        keep_mapnums (bool): Whether to keep map numbers or not
        combine_enantiomers (bool): Whether to combine enantiomers
        return_mapped (bool): Whether to additionally return atom mapped SMILES strings

    Returns:
        (list, str (optional)): Returns list of outcomes. If `return_mapped` is True,
            additionally return atom mapped SMILES strings
    """
    # New: reset atom map numbers for templates in case they have been overwritten
    # by previous uses of this template!
    rxn.reset()

    ###############################################################################
    # Run naive RDKit on ACHIRAL version of molecules
    outcomes: Tuple[Any, ...] = rxn.rxn.RunReactants((reactants.reactants_achiral,))
    if not outcomes:
        if return_mapped:
            return [], {}
        else:
            return []

    # Deduplicate outer tuple of outcomes from RDKit. The raw RunReactants output can
    # contain duplicate product tuples.
    def _outcome_key(outcome: Any) -> Tuple[str, ...]:
        # An outcome is a tuple of product mols; make the key order-independent.
        return tuple(sorted(Chem.MolToSmiles(m) for m in outcome))

    seen_outcomes = set()
    deduped_outcomes = []
    for outcome in outcomes:
        key = _outcome_key(outcome)
        if key in seen_outcomes:
            continue
        seen_outcomes.add(key)
        deduped_outcomes.append(outcome)
    outcomes = tuple(deduped_outcomes)

    # If both reactants and template are achiral, we can return the outcomes directly
    # TODO: handle intramolecular reactions, return mapping
    # also handle keep mapnums
    if not reactants.reactants_is_chiral and not rxn.template_is_chiral:
        if len(outcomes) == 1:
            if len([ele for ele in outcomes[0]]) == 1:
                achiral_outcome = outcomes[0][0]
                if not keep_mapnums:
                    for a in achiral_outcome.GetAtoms():
                        a.SetAtomMapNum(0)
                else:
                    unmapped = 900
                    for a in achiral_outcome.GetAtoms():
                        if a.GetAtomMapNum() == 0:
                            a.SetAtomMapNum(unmapped)
                            unmapped += 1
                achiral_outcome_smiles = Chem.MolToSmiles(achiral_outcome)

                if return_mapped:
                    return [achiral_outcome_smiles], {}
                else:
                    return [achiral_outcome_smiles]

    ###############################################################################

    ###############################################################################
    # Initialize, now that there is at least one outcome

    final_outcomes = set()
    mapped_outcomes = {}

    ###############################################################################
    for outcome in outcomes:
        smiles_new, mapped_info = handle_outcomes(
            outcome, reactants, rxn, keep_mapnums, return_mapped
        )
        if smiles_new is None:
            continue
        if mapped_info is None:
            continue

        final_outcomes.add(smiles_new)
        mapped_outcomes[smiles_new] = mapped_info
    ###############################################################################
    # One last fix for consolidating multiple stereospecified products...
    if combine_enantiomers:
        final_outcomes = combine_enantiomers_into_racemic(final_outcomes)
    ###############################################################################
    if return_mapped:
        return list(final_outcomes), mapped_outcomes
    else:
        return list(final_outcomes)


def handle_outcomes(
    outcome: List[Chem.Mol],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
    keep_mapnums: bool,
    return_mapped: bool,
) -> Tuple[str, Tuple[Any, ...]] | Tuple[None, None]:
    # We need to keep track of what map numbers correspond to which atoms
    # note: all reactant atoms must be mapped, so this is safe
    atoms_r = reactants.atoms_r

    # Copy reaction template so we can play around with map numbers
    template_r = rxn.template_r

    # Get molAtomMapNum->atom dictionary for template reactants and products
    atoms_rt_map = rxn.atoms_rt_map
    # TODO: cannot change atom map numbers in atoms_rt permanently?
    atoms_pt_map = rxn.atoms_pt_map

    atoms_rt = assign_outcome_atom_mapnums(outcome, reactants, atoms_rt_map)

    if validate_chiral_match(atoms_rt, atoms_r, reactants, rxn):
        return None, None

    outcomes_were_merged = False
    if len(outcome) > 1:
        merged_outcome = merge_outcomes_intramolecular(outcome)
        outcomes_were_merged = True
    else:
        merged_outcome = outcome[0]

    atoms_pt, atoms_p = assign_pt_mapnums(merged_outcome, atoms_pt_map)

    merged_outcome, bonds_were_added = check_missing_bonds(
        merged_outcome, reactants, template_r, atoms_rt, atoms_p
    )

    ###############################################################################

    # Now that we've fixed any bonds, connectivity is set. This is a good time
    # to update the property cache, since all that is left is fixing atom/bond
    # stereochemistry.
    if outcomes_were_merged or bonds_were_added:
        merged_outcome = sanitize_mol(merged_outcome)
        if merged_outcome is None:
            return None, None

    tetra_copied_from_reactants: List[Chem.Atom] = []
    if reactants.reactants_has_tetra_stereo or rxn.template_has_tetra_stereo:
        merged_outcome, tetra_copied_from_reactants = fix_tetra_stereo(
            merged_outcome, atoms_rt, atoms_r, atoms_pt
        )

        if validate_tetra_not_destroyed(merged_outcome, tetra_copied_from_reactants):
            return None, None

    if reactants.reactants_has_doublebond_stereo or rxn.template_has_doublebond_stereo:
        merged_outcome = fix_double_bond_stereochemistry(merged_outcome, reactants, rxn)

    ###############################################################################

    if return_mapped:
        # Keep track of the reacting atoms for later use in grouping
        atoms_diff = {x: atoms_are_different(atoms_r[x], atoms_p[x]) for x in atoms_rt}
        # make tuple of changed atoms
        atoms_changed = tuple([x for x in atoms_diff.keys() if atoms_diff[x]])
        mapped_outcome = Chem.MolToSmiles(merged_outcome, True)
    else:
        mapped_outcome = None
        atoms_changed = None

    if not keep_mapnums:
        for a in merged_outcome.GetAtoms():
            a.SetAtomMapNum(0)

    smiles_new = Chem.MolToSmiles(merged_outcome, isomericSmiles=True, canonical=True)
    if smiles_new is None:
        return None, None

    return (smiles_new, (mapped_outcome, atoms_changed))


def assign_outcome_atom_mapnums(
    outcome: List[Chem.Mol],
    reactants: rdchiralReactants,
    atoms_rt_map: Dict[int, Chem.Atom],
) -> Dict[int, Chem.Atom]:
    ###############################################################################
    # Look for new atoms in products that were not in
    # reactants (e.g., LGs for a retro reaction)

    idx_to_mapnum = reactants.idx_to_mapnum

    unmapped = 900
    atoms_rt: Dict[int, Chem.Atom] = {}
    for m in outcome:
        for a in m.GetAtoms():
            # Assign map number to outcome based on react_atom_idx
            if a.HasProp("react_atom_idx"):
                a.SetAtomMapNum(idx_to_mapnum(a.GetIntProp("react_atom_idx")))
                mapnum = a.GetAtomMapNum()
            else:
                mapnum = a.GetAtomMapNum()
            if not mapnum:
                mapnum = unmapped
                a.SetAtomMapNum(mapnum)
                unmapped += 1

            # Define map num -> reactant template atom map
            if a.HasProp("old_mapno"):
                rt_atom = atoms_rt_map[a.GetIntProp("old_mapno")]
                rt_atom.SetAtomMapNum(mapnum)
                atoms_rt[mapnum] = rt_atom

    return atoms_rt


def assign_pt_mapnums(
    outcome: Chem.Mol, atoms_pt_map: Dict[int, Chem.Atom]
) -> Tuple[Dict[int, Chem.Atom], Dict[int, Chem.Atom]]:
    ###############################################################################

    ###############################################################################
    # Figure out which atoms were matched in the templates
    # atoms_pt and atoms_p will be outcome-specific.
    atoms_pt = {}
    atoms_p = {}
    for a in outcome.GetAtoms():
        mapnum = a.GetAtomMapNum()
        if mapnum:
            atoms_p[mapnum] = a
        if a.HasProp("old_mapno"):
            pt_atom = atoms_pt_map[a.GetIntProp("old_mapno")]
            pt_atom.SetAtomMapNum(mapnum)
            atoms_pt[mapnum] = pt_atom

    return atoms_pt, atoms_p


def validate_chiral_match(
    atoms_rt: Dict[int, Chem.Atom],
    atoms_r: Dict[int, Chem.Atom],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
) -> bool:
    # Make sure each atom matches
    # note: this is a little weird because atom_chirality_matches takes three values,
    #       -1 (both tetra but opposite), 0 (not a match), and +1 (both tetra and match)
    #       and we only want to continue if they all equal -1 or all equal +1
    prev = None
    skip_outcome = False
    for match in (atom_chirality_matches(atoms_rt[i], atoms_r[i]) for i in atoms_rt):
        if match == 0:
            skip_outcome = True
            break
        elif match == 2:  # ambiguous case
            continue
        elif prev is None:
            prev = match
        elif match != prev:
            skip_outcome = True
            break
    if skip_outcome:
        return True

    # Check bond chirality - iterate through reactant double bonds where
    # chirality is specified (or not). atoms defined by map number
    for atoms, dirs, is_implicit in reactants.atoms_across_double_bonds:
        if all(i in atoms_rt for i in atoms):
            # All atoms definining chirality were matched to the reactant template
            # So, check if it is consistent with how the template is defined
            # ...but /=/ should match \=\ since they are both trans...
            # Convert atoms_rt to original template's atom map numbers:
            matched_atom_map_nums: Tuple[int, int, int, int] = (
                rxn.atoms_rt_idx_to_map[atoms_rt[atoms[0]].GetIdx()],
                rxn.atoms_rt_idx_to_map[atoms_rt[atoms[1]].GetIdx()],
                rxn.atoms_rt_idx_to_map[atoms_rt[atoms[2]].GetIdx()],
                rxn.atoms_rt_idx_to_map[atoms_rt[atoms[3]].GetIdx()],
            )

            if matched_atom_map_nums not in rxn.required_rt_bond_defs:
                continue  # this can happen in ring openings, for example
            dirs_template = rxn.required_rt_bond_defs[matched_atom_map_nums]
            if (
                dirs != dirs_template
                and (BondDirOpposite[dirs[0]], BondDirOpposite[dirs[1]])
                != dirs_template
                and not (dirs_template == (BondDir.NONE, BondDir.NONE) and is_implicit)
            ):
                skip_outcome = True
                break
    if skip_outcome:
        return True
    return False


def merge_outcomes_intramolecular(outcome: List[Chem.Mol]) -> Chem.Mol:
    ###############################################################################

    ###############################################################################
    # Convert product(s) to single product so that all
    # reactions can be treated as pseudo-intramolecular
    # But! check for ring openings mistakenly split into multiple
    # This can be diagnosed by duplicate map numbers (i.e., SMILES)

    mapnums = [
        a.GetAtomMapNum() for m in outcome for a in m.GetAtoms() if a.GetAtomMapNum()
    ]
    if len(mapnums) != len(set(mapnums)):  # duplicate?
        # need to do a fancy merge
        merged_mol = Chem.RWMol(outcome[0])
        merged_map_to_id = {
            a.GetAtomMapNum(): a.GetIdx()
            for a in outcome[0].GetAtoms()
            if a.GetAtomMapNum()
        }
        for j in range(1, len(outcome)):
            new_mol = outcome[j]
            for a in new_mol.GetAtoms():
                if a.GetAtomMapNum() not in merged_map_to_id:
                    merged_map_to_id[a.GetAtomMapNum()] = merged_mol.AddAtom(a)
            for b in new_mol.GetBonds():
                bi = b.GetBeginAtom().GetAtomMapNum()
                bj = b.GetEndAtom().GetAtomMapNum()
                if not merged_mol.GetBondBetweenAtoms(
                    merged_map_to_id[bi], merged_map_to_id[bj]
                ):
                    merged_mol.AddBond(
                        merged_map_to_id[bi], merged_map_to_id[bj], b.GetBondType()
                    )
                    merged_mol.GetBondBetweenAtoms(
                        merged_map_to_id[bi], merged_map_to_id[bj]
                    ).SetStereo(b.GetStereo())
                    merged_mol.GetBondBetweenAtoms(
                        merged_map_to_id[bi], merged_map_to_id[bj]
                    ).SetBondDir(b.GetBondDir())
        outcome = merged_mol.GetMol()
    else:
        new_outcome = outcome[0]
        for j in range(1, len(outcome)):
            new_outcome = rdmolops.CombineMols(new_outcome, outcome[j])
        outcome = new_outcome

    return outcome


def check_missing_bonds(
    outcome: Chem.Mol,
    reactants: rdchiralReactants,
    template_r: Chem.Mol,
    atoms_rt: Dict[int, Chem.Atom],
    atoms_p: Dict[int, Chem.Atom],
) -> Tuple[Chem.Mol, bool]:
    ###############################################################################

    ###############################################################################
    # Check for missing bonds. These are bonds that are present in the reactants,
    # not specified in the reactant template, and not in the product. Accidental
    # fragmentation can occur for intramolecular ring openings
    missing_bonds = []
    for i, j, b in reactants.bonds_by_mapnum:
        if i in atoms_p and j in atoms_p:
            # atoms from reactant bond show up in product
            if not outcome.GetBondBetweenAtoms(
                atoms_p[i].GetIdx(), atoms_p[j].GetIdx()
            ):
                # ...but there is not a bond in the product between those atoms
                if (
                    i not in atoms_rt
                    or j not in atoms_rt
                    or not template_r.GetBondBetweenAtoms(
                        atoms_rt[i].GetIdx(), atoms_rt[j].GetIdx()
                    )
                ):
                    # the reactant template did not specify a bond between those atoms (e.g., intentionally destroy)
                    missing_bonds.append((i, j, b))

    bonds_were_added = False
    if missing_bonds:
        bonds_were_added = True
        outcome = Chem.RWMol(outcome)
        rwmol_map_to_id = {
            a.GetAtomMapNum(): a.GetIdx()
            for a in outcome.GetAtoms()
            if a.GetAtomMapNum()
        }
        for i, j, b in missing_bonds:
            outcome.AddBond(rwmol_map_to_id[i], rwmol_map_to_id[j])
            new_b = outcome.GetBondBetweenAtoms(rwmol_map_to_id[i], rwmol_map_to_id[j])
            new_b.SetBondType(b.GetBondType())
            new_b.SetBondDir(b.GetBondDir())
            new_b.SetIsAromatic(b.GetIsAromatic())
        outcome = outcome.GetMol()
        atoms_p = {
            a.GetAtomMapNum(): a for a in outcome.GetAtoms() if a.GetAtomMapNum()
        }

    return outcome, bonds_were_added


def fix_tetra_stereo(
    outcome: Chem.Mol,
    atoms_rt: Dict[int, Chem.Atom],
    atoms_r: Dict[int, Chem.Atom],
    atoms_pt: Dict[int, Chem.Atom],
) -> Tuple[Chem.Mol, List[Chem.Atom]]:
    ###############################################################################
    # Correct tetra chirality in the outcome
    tetra_copied_from_reactants = []
    for a in outcome.GetAtoms():
        # Participants in reaction core (from reactants) will have old_mapno
        # Spectators present in reactants will have react_atom_idx
        # ...so new atoms will have neither!
        atom_map_num = a.GetAtomMapNum()
        if not a.HasProp("old_mapno"):
            # Not part of the reactants template

            if a.HasProp("react_atom_idx"):
                copy_chirality(atoms_r[atom_map_num], a)
                if a.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                    tetra_copied_from_reactants.append(a)

        else:
            # Part of reactants and reaction core

            if template_atom_could_have_been_tetra(atoms_rt[atom_map_num]):
                if template_atom_could_have_been_tetra(atoms_pt[atom_map_num]):
                    # Was the product template specified?

                    if (
                        atoms_pt[atom_map_num].GetChiralTag()
                        == ChiralType.CHI_UNSPECIFIED
                    ):
                        # No, leave unspecified in product
                        a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

                    else:
                        # Yes

                        # Was the reactant template specified?

                        if (
                            atoms_rt[atom_map_num].GetChiralTag()
                            == ChiralType.CHI_UNSPECIFIED
                        ):
                            # No, so the reaction introduced chirality
                            copy_chirality(atoms_pt[atom_map_num], a)

                        else:
                            # Yes, so we need to check if chirality should be preserved or inverted
                            copy_chirality(atoms_r[atom_map_num], a)
                            if (
                                atom_chirality_matches(
                                    atoms_pt[atom_map_num],
                                    atoms_rt[atom_map_num],
                                )
                                == -1
                            ):
                                a.InvertChirality()

            else:
                if not template_atom_could_have_been_tetra(atoms_pt[atom_map_num]):
                    copy_chirality(atoms_r[atom_map_num], a)
                    if a.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetra_copied_from_reactants.append(a)

                else:
                    copy_chirality(atoms_pt[atom_map_num], a)
    return outcome, tetra_copied_from_reactants


def validate_tetra_not_destroyed(
    outcome: Chem.Mol, tetra_copied_from_reactants: List[Chem.Atom]
) -> bool:

    # Now, check to see if we have destroyed chirality
    # this occurs when chirality was not actually possible (e.g., due to
    # symmetry) but we had assigned a tetrahedral center originating
    # from the reactants.
    #    ex: SMILES C(=O)1C[C@H](Cl)CCC1
    #        SMARTS [C:1]-[C;H0;D3;+0:2](-[C:3])=[O;H0;D1;+0]>>[C:1]-[CH2;D2;+0:2]-[C:3]
    if len(tetra_copied_from_reactants) > 0:
        Chem.AssignStereochemistry(outcome, cleanIt=True, force=True)
        for a in tetra_copied_from_reactants:
            if a.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                return True
    return False


def sanitize_mol(merged_outcome):
    try:
        Chem.SanitizeMol(merged_outcome)
        merged_outcome.UpdatePropertyCache()
    except ValueError:
        return None
    return merged_outcome


def fix_double_bond_stereochemistry(
    outcome: Chem.Mol, reactants: rdchiralReactants, rxn: rdchiralReaction
) -> Chem.Mol:
    ###############################################################################

    ###############################################################################
    # Correct bond directionality in the outcome
    initial_bond_dirs = bond_dirs_by_mapnum(outcome)
    for b in outcome.GetBonds():
        if b.GetBondType() != BondType.DOUBLE:
            continue

        # Ring double bonds do not need to be touched(?)
        if b.IsInRing():
            continue

        ba = b.GetBeginAtom()
        bb = b.GetEndAtom()

        # Is it possible at all to specify this bond?
        if ba.GetDegree() == 1 or bb.GetDegree() == 1:
            continue

        if ba.HasProp("old_mapno") and bb.HasProp("old_mapno"):
            # Need to rely on templates for bond chirality, both atoms were
            # in the reactant template
            if (
                ba.GetIntProp("old_mapno"),
                bb.GetIntProp("old_mapno"),
            ) in rxn.required_bond_defs_coreatoms:
                continue

        elif not ba.HasProp("react_atom_idx") and not bb.HasProp("react_atom_idx"):
            # The atoms were both created by the product template, so any bond
            # stereochemistry should have been instantiated by the product template
            # already...hopefully...otherwise it isn't specific enough?
            continue

        # Need to copy from reactants, this double bond was simply carried over,
        # *although* one of the atoms could have reacted and been an auxilliary
        # atom in the reaction, e.g., C/C=C(/CO)>>C/C=C(/C[Br])

        # Start with setting the BeginAtom
        begin_atom_specified = restore_bond_stereo_to_sp2_atom(
            ba, reactants.bond_dirs_by_mapnum
        )

        if not begin_atom_specified:
            # don't bother setting other side of bond, since we won't be able to
            # fully specify this bond as cis/trans
            continue

        # Look at other side of the bond now, the EndAtom
        _ = restore_bond_stereo_to_sp2_atom(bb, reactants.bond_dirs_by_mapnum)

    # Need to check whether a conjugated system was changed.
    if initial_bond_dirs:
        Chem.SanitizeMol(
            outcome, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        )
        outcome.UpdatePropertyCache(strict=False)
        _ = correct_conjugated(initial_bond_dirs, outcome)
    return outcome
