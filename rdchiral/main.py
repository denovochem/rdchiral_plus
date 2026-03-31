from __future__ import print_function

from typing import Any, Dict, List, Optional, Tuple, Union

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
) -> Any:
    """
    Run a reaction by constructing `rdchiralReaction` and `rdchiralReactants` from text inputs.

    Args:
        reaction_smarts (str): Reaction SMARTS string used to initialize `rdchiralReaction`.
        reactant_smiles (str): Reactant SMILES string used to initialize `rdchiralReactants`.
        custom_reactant_mapping (bool): If True, assume the input reactants already contain
            an atom-mapping that should be preserved/used.
        keep_mapnums (bool): If True, preserve atom-map numbers in returned product SMILES.
        combine_enantiomers (bool): If True, attempt to combine enantiomeric outcomes into
            racemic outcomes.
        return_mapped (bool): If True, also return per-outcome atom-mapped information.

    Returns:
        Union[List[str], Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]:
            - If `return_mapped` is False: A list of product SMILES strings.
            - If `return_mapped` is True: A tuple of `(outcomes, mapped_outcomes)` as
              returned by `rdchiralRun`.

    Note:
        This helper is convenient for one-off use but is not recommended for batch/library
        workflows because template/reactant initialization is relatively expensive. For
        repeated application, construct `rdchiralReaction` and `rdchiralReactants` once and
        call `rdchiralRun` directly.
    """
    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles, custom_reactant_mapping)
    return rdchiralRun(
        rxn=rxn,
        reactants=reactants,
        keep_mapnums=keep_mapnums,
        combine_enantiomers=combine_enantiomers,
        return_mapped=return_mapped,
        skip_reset=True,
    )


def rdchiralRun(
    rxn: rdchiralReaction,
    reactants: rdchiralReactants,
    keep_mapnums: bool = False,
    combine_enantiomers: bool = True,
    return_mapped: bool = False,
    skip_reset: bool = False,
) -> Any:
    """
    Apply a pre-initialized `rdchiralReaction` template to pre-initialized reactants.

    Args:
        rxn (rdchiralReaction): Reaction template wrapper, including an RDKit reaction
            object and precomputed stereochemistry/atom-map bookkeeping.
        reactants (rdchiralReactants): Reactants wrapper, including an RDKit molecule
            and stereochemistry/atom-map bookkeeping.
        keep_mapnums (bool): If False, clear atom-map numbers from returned product
            SMILES. If True, preserve map numbers; atoms that are unmapped in the
            product may be assigned new map numbers (implementation-dependent).
        combine_enantiomers (bool): If True, attempt to combine enantiomeric outcomes
            into racemic outcomes.
        return_mapped (bool): If True, also return per-outcome atom-mapped information.
        skip_reset (bool): If True, skip resetting the reaction object before running.

    Returns:
        Any:
            - If `return_mapped` is False: A list of product SMILES strings.
            - If `return_mapped` is True: A tuple of `(outcomes, mapped_outcomes)`.
              `outcomes` is the list of product SMILES strings. `mapped_outcomes` maps
              each product SMILES string to `(mapped_smiles, atoms_changed)`, where
              `mapped_smiles` is the mapped SMILES for that product and
              `atoms_changed` is a tuple of atom-map numbers whose corresponding atoms
              differ between reactants and products.

    Note:
        This function mutates `rxn` via `rxn.reset()` to restore template atom-map
        numbers, and may mutate intermediate RDKit molecules produced by RDKit during
        post-processing.
    """
    if not rxn.fast_reactant_smarts or not reactants.fast_reactants:
        if return_mapped:
            return [], {}
        else:
            return []

    if not reactants.reactants_achiral.HasSubstructMatch(
        rxn.fast_reactant_smarts, useChirality=False
    ):
        if return_mapped:
            return [], {}
        else:
            return []

    if not skip_reset:
        rxn.reset()

    # Run naive RDKit on achiral version of molecules
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...] = rxn.rxn.RunReactants(
        (reactants.reactants_achiral,)
    )
    if not outcomes:
        if return_mapped:
            return [], {}
        else:
            return []

    outcomes = deduplicate_outcomes(outcomes, reactants, rxn)

    result = return_non_stereo_outcome_early(
        outcomes, reactants, rxn, return_mapped, keep_mapnums
    )
    if result is not None:
        return result

    final_outcomes = set()
    mapped_outcomes = {}
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

    if combine_enantiomers:
        final_outcomes = combine_enantiomers_into_racemic(final_outcomes)

    if return_mapped:
        return list(final_outcomes), mapped_outcomes
    else:
        return list(final_outcomes)


def deduplicate_outcomes_with_smiles(
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...],
) -> Tuple[Tuple[Chem.Mol, ...], ...]:
    """
    Deduplicate RDKit reaction outcomes using canonical product SMILES.

    Args:
        outcomes (Tuple[Tuple[Chem.Mol, ...], ...]): Raw output from
            `ChemicalReaction.RunReactants`, represented as a tuple of outcomes where
            each outcome is a tuple of product molecules.

    Returns:
        Tuple[Tuple[Chem.Mol, ...], ...]: The input `outcomes` with duplicate outcome
        tuples removed. Two outcomes are considered duplicates if the sorted tuple of
        canonical SMILES for their product molecules is identical.
    """

    def _outcome_key(outcome: Any) -> Tuple[str, ...]:
        return tuple(sorted(Chem.MolToSmiles(m, canonical=True) for m in outcome))

    if len(outcomes) > 1:
        seen_outcomes = set()
        deduped_outcomes = []
        for outcome in outcomes:
            key = _outcome_key(outcome)
            if key in seen_outcomes:
                continue
            seen_outcomes.add(key)
            deduped_outcomes.append(outcome)
        outcomes = tuple(deduped_outcomes)
    return outcomes


def deduplicate_outcomes(
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
) -> Tuple[Tuple[Chem.Mol, ...], ...]:
    """
    Deduplicate RDKit reaction outcomes by identical reactant substructure matches.

    This is an alternative to `deduplicate_outcomes_with_smiles` that avoids SMILES
    generation. Instead, it computes all substructure matches of the reaction's
    reactant template against the (achiral) reactants with `uniquify=False` and
    removes outcomes that correspond to repeated matches.

    Args:
        outcomes (Tuple[Tuple[Chem.Mol, ...], ...]): Raw output from
            `ChemicalReaction.RunReactants`, represented as a tuple of outcomes where
            each outcome is a tuple of product molecules.
        reactants (rdchiralReactants): Initialized reactants container. The
            `reactants_achiral` molecule is used for substructure matching.
        rxn (rdchiralReaction): Initialized reaction container. The `template_r`
            reactant template is matched against `reactants.reactants_achiral`.

    Returns:
        Tuple[Tuple[Chem.Mol, ...], ...]: The input `outcomes` with duplicates removed
        such that at most one outcome is kept for each unique reactant substructure
        match.

    Note:
        This function assumes that the order of `outcomes` corresponds to the order
        of `GetSubstructMatches(rxn.template_r, uniquify=False)`.
    """
    if len(outcomes) > 1:
        substruct_matches = reactants.reactants_achiral.GetSubstructMatches(
            rxn.template_r, uniquify=False
        )
        sorted_substruct_matches = tuple(
            sorted(substruct_match for substruct_match in substruct_matches)
        )

        seen_outcomes = set()
        deduped_outcomes = []
        for outcome, sorted_substruct_match in zip(outcomes, sorted_substruct_matches):
            if sorted_substruct_match in seen_outcomes:
                continue
            seen_outcomes.add(sorted_substruct_match)
            deduped_outcomes.append(outcome)
        outcomes = tuple(deduped_outcomes)
    return outcomes


def return_non_stereo_outcome_early(
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
    return_mapped: bool = False,
    keep_mapnums: bool = False,
) -> Optional[
    Union[List[str], Tuple[List[str], Dict[str, Tuple[Tuple[int, ...], ...]]]]
]:
    """
    Return a non-stereochemical outcome early when both reactants and template are achiral.

    This helper is an optimization used by the main product enumeration logic. When both the
    input reactants and the reaction template are achiral, and the reaction application
    produced exactly one outcome containing exactly one product molecule, the product can be
    returned directly without stereochemistry handling.

    Args:
        outcomes (Tuple[Tuple[Chem.Mol, ...], ...]): Outcomes returned by reaction application,
            where each outer tuple element is an outcome containing one or more product
            molecules.
        reactants (rdchiralReactants): Reactant container providing achiral reactant molecule
            and mapping information.
        rxn (rdchiralReaction): Reaction/template container providing the reactant-side
            template used for substructure matching.
        return_mapped (bool): If True, also return a mapping dictionary keyed by the mapped
            product SMILES with values containing the (non-unique) substructure matches of the
            achiral reactants to the reactant-side template.
        keep_mapnums (bool): If True, keep atom mapping numbers in the returned product SMILES.
            Unmapped atoms (atom map number 0) are assigned new map numbers starting at 900.

    Returns:
        Optional[Union[List[str], Tuple[List[str], Dict[str, Tuple[Tuple[int, ...], ...]]]]]:
            If the early-return conditions are not met, returns None. Otherwise returns a list
            containing a single product SMILES string. If `return_mapped` is True, returns a
            tuple of:
            - The product SMILES list.
            - A dict mapping mapped product SMILES to the reactant substructure matches.

    Note:
        If `keep_mapnums` is False, this function clears atom map numbers in-place on the
        product molecule before converting to SMILES.
    """
    if reactants.reactants_is_chiral or rxn.template_is_chiral:
        return None

    # TODO: remove this guard
    if len(outcomes) != 1:
        return None

    # TODO: remove this guard
    if return_mapped or keep_mapnums:
        return None

    if set([len(outcome) for outcome in outcomes]) != {1}:
        return None

    final_outcomes_list: List[str] = []
    mapped_outcomes_dict: Dict[str, Tuple[Tuple[int, ...], ...]] = {}

    for outcome in outcomes:
        if keep_mapnums:
            mapped_outcome = Chem.Mol(outcome[0])
            unmapped = 900
            for a in mapped_outcome.GetAtoms():
                if a.GetAtomMapNum() == 0:
                    a.SetAtomMapNum(unmapped)
                unmapped += 1
            mapped_outcome_smiles = Chem.MolToSmiles(mapped_outcome)
            final_outcomes_list.append(mapped_outcome_smiles)
            continue

        for a in outcome[0].GetAtoms():
            a.SetAtomMapNum(0)
        unmapped_outcome_smiles = Chem.MolToSmiles(outcome[0])
        final_outcomes_list.append(unmapped_outcome_smiles)

    if return_mapped:
        substruct_matches = reactants.reactants_achiral.GetSubstructMatches(
            rxn.template_r, uniquify=False
        )

        for final_outcome, substruct_match in zip(
            final_outcomes_list, substruct_matches
        ):
            mapped_outcomes_dict[final_outcome] = (substruct_match,)

        return final_outcomes_list, mapped_outcomes_dict
    else:
        return final_outcomes_list


def handle_outcomes(
    outcome: Tuple[Chem.Mol, ...],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
    keep_mapnums: bool,
    return_mapped: bool,
) -> Union[Tuple[str, Tuple[Any, ...]], Tuple[None, None]]:
    """
    Post-process a single raw RDKit reaction outcome into a final product SMILES.

    This function aligns/repairs atom-map numbers in the product(s), optionally merges
    multi-product outcomes into a single intramolecular product, restores any bonds
    missing from the reaction outcome, and fixes stereochemistry (tetrahedral and
    double-bond) to be consistent with the input reactants and reaction template.

    Args:
        outcome (Tuple[Chem.Mol, ...]): One outcome from RDKit reaction application,
            represented as a tuple of product molecules.
        reactants (rdchiralReactants): Initialized reactants container, including
            atom-map bookkeeping and stereo flags.
        rxn (rdchiralReaction): Initialized reaction container, including reactant and
            product templates and atom-map lookup tables.
        keep_mapnums (bool): If False, clear all atom-map numbers from the final
            product molecule before generating the returned SMILES.
        return_mapped (bool): If True, also return a mapped SMILES for the final
            product and the tuple of atom-map numbers that changed between reactants
            and products.

    Returns:
        Union[Tuple[str, Tuple[Any, ...]], Tuple[None, None]]:
            - On success: A tuple of `(smiles_new, (mapped_outcome, atoms_changed))`.
              `smiles_new` is a canonical isomeric SMILES for the final product.
              If `return_mapped` is True, `mapped_outcome` is the mapped SMILES for the
              final product and `atoms_changed` is a tuple of atom-map numbers whose
              corresponding atoms differ between reactants and products; otherwise both
              values are None.
            - On failure: `(None, None)` if the outcome is rejected by chiral/stereo
              validation, sanitization fails after connectivity repair, or SMILES
              generation fails.

    Note:
        This function mutates the RDKit molecule(s) in `outcome` (e.g., atom-map
        numbers, bond connectivity, and stereochemistry) as part of the repair and
        standardization process.
    """
    # Get molAtomMapNum->atom dictionary for template reactants and products
    atoms_rt_map = rxn.atoms_rt_map
    outcome, atoms_rt, atoms_rt_map = assign_outcome_atom_mapnums(
        outcome, reactants, atoms_rt_map
    )

    # We need to keep track of what map numbers correspond to which atoms
    # note: all reactant atoms must be mapped, so this is safe
    atoms_r = reactants.atoms_r
    if validate_chiral_match(atoms_rt, atoms_r, reactants, rxn):
        return None, None

    outcomes_were_merged = False
    if len(outcome) > 1:
        merged_outcome = merge_outcomes_intramolecular(outcome)
        outcomes_were_merged = True
    else:
        merged_outcome = outcome[0]

    # TODO: cannot change atom map numbers in atoms_rt permanently?
    atoms_pt_map = rxn.atoms_pt_map
    atoms_pt, atoms_p, atoms_pt_map = assign_pt_mapnums(merged_outcome, atoms_pt_map)

    # Copy reaction template so we can play around with map numbers
    template_r = rxn.template_r
    merged_outcome, atoms_p, bonds_were_added = check_missing_bonds(
        merged_outcome, reactants, template_r, atoms_rt, atoms_p
    )

    # Now that we've fixed any bonds, connectivity is set. This is a good time
    # to update the property cache, since all that is left is fixing atom/bond
    # stereochemistry.
    if outcomes_were_merged or bonds_were_added:
        sanitized_outcome = sanitize_mol(merged_outcome)
        if sanitized_outcome is None:
            return None, None
        merged_outcome = sanitized_outcome

    tetra_copied_from_reactants: List[Chem.Atom] = []
    if reactants.reactants_has_tetra_stereo or rxn.template_has_tetra_stereo:
        merged_outcome, tetra_copied_from_reactants = fix_tetra_stereo(
            merged_outcome, atoms_rt, atoms_r, atoms_pt
        )

        if validate_tetra_not_destroyed(merged_outcome, tetra_copied_from_reactants):
            return None, None

    if reactants.reactants_has_doublebond_stereo or rxn.template_has_doublebond_stereo:
        merged_outcome = fix_double_bond_stereochemistry(merged_outcome, reactants, rxn)

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
    outcome: Tuple[Chem.Mol, ...],
    reactants: rdchiralReactants,
    atoms_rt_map: Dict[int, Chem.Atom],
) -> Tuple[Tuple[Chem.Mol, ...], Dict[int, Chem.Atom], Dict[int, Chem.Atom]]:
    """
    Assign consistent atom-map numbers to all atoms in an outcome and build a reactant-template lookup keyed by those map numbers.

    Args:
        outcome (Tuple[Chem.Mol, ...]): Tuple of product molecules produced by template application. Atoms may have a
            `react_atom_idx` property indicating the corresponding reactant atom index, and/or an `old_mapno` property
            indicating the original reactant-template atom-map number.
        reactants (rdchiralReactants): Reactant container providing an `idx_to_mapnum` mapping from reactant atom index
            to the atom-map number used for the reactants.
        atoms_rt_map (Dict[int, Chem.Atom]): Mapping from original reactant-template atom-map number (as referenced by
            the `old_mapno` property) to the corresponding reactant-template atom.

    Returns:
        Tuple[Tuple[Chem.Mol, ...], Dict[int, Chem.Atom], Dict[int, Chem.Atom]]:
            - First element: The (mutated) `outcome` tuple, with atom-map numbers assigned/filled in.
            - Second element: Mapping from outcome atom-map number to the matched reactant-template atom.
            - Third element: The (mutated) `atoms_rt_map` mapping that was passed in.

    Note:
        If an outcome atom has `react_atom_idx`, its map number is overwritten using `reactants.idx_to_mapnum`. Any atom
        lacking a map number is assigned a new one starting at 900. This function also mutates reactant-template atoms in
        `atoms_rt_map` by calling `SetAtomMapNum` when matching via `old_mapno`.
    """
    idx_to_mapnum = reactants.idx_to_mapnum

    unmapped = 900
    atoms_rt: Dict[int, Chem.Atom] = {}
    for m in outcome:
        for a in m.GetAtoms():
            # Assign map number to outcome based on react_atom_idx
            if a.HasProp("react_atom_idx"):
                mapnum = idx_to_mapnum(a.GetIntProp("react_atom_idx"))
                if not mapnum:
                    mapnum = unmapped
                    unmapped += 1
                a.SetAtomMapNum(mapnum)
            else:
                mapnum = a.GetAtomMapNum()
                if not mapnum:
                    mapnum = unmapped
                    a.SetAtomMapNum(mapnum)
                    unmapped += 1

            # Define map num -> reactant template atom map
            if a.HasProp("old_mapno"):
                old_mapno = a.GetIntProp("old_mapno")
                rt_atom = atoms_rt_map.get(old_mapno)
                if rt_atom is not None:
                    rt_atom.SetAtomMapNum(mapnum)
                    atoms_rt[mapnum] = rt_atom

    return outcome, atoms_rt, atoms_rt_map


def assign_pt_mapnums(
    outcome: Chem.Mol, atoms_pt_map: Dict[int, Chem.Atom]
) -> Tuple[Dict[int, Chem.Atom], Dict[int, Chem.Atom], Dict[int, Chem.Atom]]:
    """
    Build product-template and product-atom lookup dicts keyed by atom-map number for a single reaction outcome.

    Args:
        outcome (Chem.Mol): Product molecule generated by applying the reaction template. Atoms in this molecule may
            have an `old_mapno` property that references the original atom-map number in the product template.
        atoms_pt_map (Dict[int, Chem.Atom]): Mapping from original product-template atom-map number to the
            corresponding product-template atom.

    Returns:
        Tuple[Dict[int, Chem.Atom], Dict[int, Chem.Atom], Dict[int, Chem.Atom]]:
            - First dict: Mapping from outcome atom-map number to the matched product-template atom, with the
              template atom's atom-map number updated to the outcome atom-map number.
            - Second dict: Mapping from outcome atom-map number to the corresponding atom in `outcome`.
            - Third dict: The (mutated) `atoms_pt_map` mapping that was passed in.

    Note:
        This function mutates the `Chem.Atom` objects in `atoms_pt_map` by calling `SetAtomMapNum` on any
        product-template atoms matched via the `old_mapno` property.
    """
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

    return atoms_pt, atoms_p, atoms_pt_map


def validate_chiral_match(
    atoms_rt: Dict[int, Chem.Atom],
    atoms_r: Dict[int, Chem.Atom],
    reactants: rdchiralReactants,
    rxn: rdchiralReaction,
) -> bool:
    """
    Validate whether the chirality constraints in the matched reactants are consistent with the reaction template.

    Args:
        atoms_rt (Dict[int, Chem.Atom]): Mapping from atom-map number to the matched reactant-template atom.
        atoms_r (Dict[int, Chem.Atom]): Mapping from atom-map number to the corresponding atom in the input reactants.
        reactants (rdchiralReactants): Reactant container providing precomputed stereochemical features for the input
            reactants (e.g., atoms across specified double bonds).
        rxn (rdchiralReaction): Reaction/template container providing required reactant-template bond direction
            definitions and index-to-map-number mappings.

    Returns:
        bool: True if the outcome should be skipped due to a stereochemical mismatch; False if the stereochemistry is
            compatible.

    Note:
        Atom chirality is checked by requiring that all matched tetrahedral centers either match with the same sign
        (all +1) or are all inverted (all -1) according to `atom_chirality_matches`. A return value of 2 from
        `atom_chirality_matches` is treated as "no constraint" and ignored.

        Double-bond directionality is checked for any matched 4-atom tuples in `reactants.atoms_across_double_bonds`.
        A bond definition is considered compatible if it matches the template directions directly, matches after
        applying `BondDirOpposite` to both directions (trans equivalence), or if the template specifies
        `(BondDir.NONE, BondDir.NONE)` and the reactant stereochemistry was implicit.
    """
    prev: Optional[int] = None
    skip_outcome = False
    for i in atoms_rt:
        match: int = atom_chirality_matches(atoms_rt[i], atoms_r[i])
        if match == 0:
            skip_outcome = True
            break
        elif match == 2:
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
    atoms_rt_idx_to_map = rxn.atoms_rt_idx_to_map
    required_rt_bond_defs = rxn.required_rt_bond_defs

    for atoms, dirs, is_implicit in reactants.atoms_across_double_bonds:
        if all(i in atoms_rt for i in atoms):
            # All atoms definining chirality were matched to the reactant template
            # So, check if it is consistent with how the template is defined
            # ...but /=/ should match \=\ since they are both trans...
            # Convert atoms_rt to original template's atom map numbers:
            matched_atom_map_nums: Tuple[int, int, int, int] = (
                atoms_rt_idx_to_map[atoms_rt[atoms[0]].GetIdx()],
                atoms_rt_idx_to_map[atoms_rt[atoms[1]].GetIdx()],
                atoms_rt_idx_to_map[atoms_rt[atoms[2]].GetIdx()],
                atoms_rt_idx_to_map[atoms_rt[atoms[3]].GetIdx()],
            )

            if matched_atom_map_nums not in required_rt_bond_defs:
                continue  # this can happen in ring openings, for example
            dirs_template = required_rt_bond_defs[matched_atom_map_nums]
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


def merge_outcomes_intramolecular(outcome: Tuple[Chem.Mol, ...]) -> Chem.Mol:
    """
    Merge a tuple of product molecules into a single product molecule for pseudo-intramolecular handling.

    Args:
        outcome (Tuple[Chem.Mol, ...]): Tuple of product molecules produced by applying a reaction.

    Returns:
        Chem.Mol: A single merged product molecule.

    Note:
        If the products contain duplicate atom-map numbers, this function performs a map-number-based merge
        (intended to handle ring-opening cases that were mistakenly split into multiple fragments) and
        copies bond type, stereo, and bond direction when adding missing bonds. Otherwise, it merges
        products using `rdmolops.CombineMols`.
    """
    mapnums = []
    merged_map_to_id = {}
    for i, m in enumerate(outcome):
        for a in m.GetAtoms():
            mapnum = a.GetAtomMapNum()
            if not mapnum:
                continue
            mapnums.append(mapnum)
            if i == 0:
                merged_map_to_id[mapnum] = a.GetIdx()
    if len(mapnums) != len(set(mapnums)):  # duplicate?
        # need to do a fancy merge
        merged_mol = Chem.RWMol(outcome[0])
        for j in range(1, len(outcome)):
            new_mol = outcome[j]
            for a in new_mol.GetAtoms():
                map_num = a.GetAtomMapNum()
                if map_num not in merged_map_to_id:
                    merged_map_to_id[map_num] = merged_mol.AddAtom(a)
            for b in new_mol.GetBonds():
                bi = b.GetBeginAtom().GetAtomMapNum()
                bj = b.GetEndAtom().GetAtomMapNum()
                bi_bj_bond = merged_mol.GetBondBetweenAtoms(
                    merged_map_to_id[bi], merged_map_to_id[bj]
                )
                if not bi_bj_bond:
                    merged_mol.AddBond(
                        merged_map_to_id[bi], merged_map_to_id[bj], b.GetBondType()
                    )
                    newly_added_bond = merged_mol.GetBondBetweenAtoms(
                        merged_map_to_id[bi], merged_map_to_id[bj]
                    )
                    newly_added_bond.SetStereo(b.GetStereo())
                    newly_added_bond.SetBondDir(b.GetBondDir())
        merged_outcome = merged_mol.GetMol()
    else:
        merged_outcome = outcome[0]
        for j in range(1, len(outcome)):
            merged_outcome = rdmolops.CombineMols(merged_outcome, outcome[j])

    return merged_outcome


def check_missing_bonds(
    outcome: Chem.Mol,
    reactants: rdchiralReactants,
    template_r: Chem.Mol,
    atoms_rt: Dict[int, Chem.Atom],
    atoms_p: Dict[int, Chem.Atom],
) -> Tuple[Chem.Mol, Dict[int, Chem.Atom], bool]:
    """
    Re-add reactant bonds to the outcome when they were omitted by the template and would otherwise fragment the product.

    This function identifies bonds that are present in the original reactants and
    connect two mapped atoms that also appear in the product, but for which the
    current `outcome` lacks a bond between the corresponding product atoms. Such a
    bond is considered "missing" only if it was also not specified in the reactant
    template (`template_r`), which indicates the bond was not intentionally broken
    by the reaction definition.

    Args:
        outcome (Chem.Mol): Current product molecule to inspect and potentially
            modify.
        reactants (rdchiralReactants): Reactants container providing the original
            reactant bonds keyed by atom-map number.
        template_r (Chem.Mol): Reactant template molecule used to determine
            whether a bond was intentionally specified (and thus potentially
            intentionally broken).
        atoms_rt (Dict[int, Chem.Atom]): Mapping from atom-map number to the
            corresponding atom in the reactant template.
        atoms_p (Dict[int, Chem.Atom]): Mapping from atom-map number to the
            corresponding atom in the product `outcome`.

    Returns:
        Tuple[Chem.Mol, Dict[int, Chem.Atom], bool]:
            - Chem.Mol: The (possibly modified) outcome molecule.
            - Dict[int, Chem.Atom]: Updated mapping of atom-map number to atoms in
              the returned outcome molecule.
            - bool: True if one or more missing bonds were added, otherwise False.

    Note:
        When adding a missing bond, this function copies bond type, bond
        direction, and aromaticity from the corresponding reactant bond.
    """
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
        rwmol_map_to_id = {}

        for a in outcome.GetAtoms():
            atom_map_num = a.GetAtomMapNum()
            if atom_map_num:
                rwmol_map_to_id[atom_map_num] = a.GetIdx()

        for i, j, b in missing_bonds:
            outcome.AddBond(rwmol_map_to_id[i], rwmol_map_to_id[j])
            new_b = outcome.GetBondBetweenAtoms(rwmol_map_to_id[i], rwmol_map_to_id[j])
            new_b.SetBondType(b.GetBondType())
            new_b.SetBondDir(b.GetBondDir())
            new_b.SetIsAromatic(b.GetIsAromatic())

        outcome = outcome.GetMol()

        for a in outcome.GetAtoms():
            atom_map_num = a.GetAtomMapNum()
            if atom_map_num:
                atoms_p[atom_map_num] = a

    return outcome, atoms_p, bonds_were_added


def fix_tetra_stereo(
    outcome: Chem.Mol,
    atoms_rt: Dict[int, Chem.Atom],
    atoms_r: Dict[int, Chem.Atom],
    atoms_pt: Dict[int, Chem.Atom],
) -> Tuple[Chem.Mol, List[Chem.Atom]]:
    """
    Correct tetrahedral chirality annotations in a merged outcome molecule.

    For each atom in `outcome`, this function determines whether tetrahedral
    chirality should be copied from the original reactants, left unspecified, or
    taken from the product template. For atoms in the reaction core where both
    reactant and product templates could encode tetrahedral stereochemistry, the
    outcome chirality is preserved or inverted based on whether the templates
    indicate retention or inversion.

    Args:
        outcome (Chem.Mol): Product molecule to modify in-place.
        atoms_rt (Dict[int, Chem.Atom]): Reactant-template atoms keyed by atom-map
            number.
        atoms_r (Dict[int, Chem.Atom]): Original reactant atoms keyed by atom-map
            number.
        atoms_pt (Dict[int, Chem.Atom]): Product-template atoms keyed by atom-map
            number.

    Returns:
        Tuple[Chem.Mol, List[Chem.Atom]]:
            - Chem.Mol: The same `outcome` molecule instance after tetrahedral
              chirality updates.
            - List[Chem.Atom]: Atoms in `outcome` whose tetrahedral chirality was
              copied from the reactants (used for later validation).
    """
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

        ## TODO reorder this so that getchiraltag is called first? its cheaper?
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
    """
    Check whether any tetrahedral stereocenter copied from the reactants became unspecified in the product.

    Args:
        outcome (Chem.Mol): Product molecule to validate after applying the reaction/template.
        tetra_copied_from_reactants (List[Chem.Atom]): Atoms in `outcome` whose tetrahedral chirality was
            copied/propagated from the reactants.

    Returns:
        bool: True if stereochemistry assignment indicates at least one copied tetrahedral center has
            `CHI_UNSPECIFIED` in `outcome` (i.e., chirality was effectively destroyed or was not actually
            possible); otherwise False.
    """
    if len(tetra_copied_from_reactants) > 0:
        Chem.AssignStereochemistry(outcome, cleanIt=True, force=True)
        for a in tetra_copied_from_reactants:
            if a.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                return True
    return False


def sanitize_mol(merged_outcome: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Sanitize a merged outcome molecule using RDKit and return `None` on sanitization failure.

    Args:
        merged_outcome (Chem.Mol): The molecule to sanitize in-place.

    Returns:
        Optional[Chem.Mol]: The same `merged_outcome` instance after successful sanitization and
            property-cache update, or `None` if RDKit sanitization raises a `ValueError`.
    """
    try:
        Chem.SanitizeMol(merged_outcome)
        merged_outcome.UpdatePropertyCache()
    except ValueError:
        return None
    return merged_outcome


def fix_double_bond_stereochemistry(
    outcome: Chem.Mol, reactants: rdchiralReactants, rxn: rdchiralReaction
) -> Chem.Mol:
    """
    Restore or preserve double-bond stereochemistry in a predicted product molecule.

    This function iterates over non-ring double bonds in `outcome` and attempts to
    ensure that bond-direction annotations needed for E/Z specification are
    consistent with the input `reactants` and the reaction template constraints in
    `rxn`. When a double bond was carried over from the reactants but lost bond
    directionality during reaction application/merging, bond directions are
    restored on the adjacent atoms when possible. If the bond stereochemistry is
    fully specified by the reaction templates (or the bond was created entirely by
    the product template), it is left unchanged.

    Args:
        outcome (Chem.Mol): Product molecule whose double-bond stereo annotations
            may need to be corrected.
        reactants (rdchiralReactants): Reactant container providing original bond
            directionality keyed by atom-map number.
        rxn (rdchiralReaction): Reaction template information, including required
            bond definitions on core atoms.

    Returns:
        Chem.Mol: The (potentially modified) `outcome` molecule with updated bond
            directionality and conjugation state.

    Note:
        This function mutates and returns the same `outcome` instance.
        Ring double bonds and terminal double bonds (degree 1 at either end) are
        skipped. After attempting to restore bond directions, conjugation is
        re-perceived and conjugated systems are corrected using the initial bond
        directions captured at the start of the function.
    """
    bond_dirs_by_map_num = reactants.bond_dirs_by_mapnum
    required_bond_defs_coreatoms = rxn.required_bond_defs_coreatoms
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
            ) in required_bond_defs_coreatoms:
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
        begin_atom_specified = restore_bond_stereo_to_sp2_atom(ba, bond_dirs_by_map_num)

        if not begin_atom_specified:
            # don't bother setting other side of bond, since we won't be able to
            # fully specify this bond as cis/trans
            continue

        # Look at other side of the bond now, the EndAtom
        _ = restore_bond_stereo_to_sp2_atom(bb, bond_dirs_by_map_num)

    # Need to check whether a conjugated system was changed.
    if initial_bond_dirs:
        Chem.SanitizeMol(
            outcome, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        )
        outcome.UpdatePropertyCache(strict=False)
        _ = correct_conjugated(initial_bond_dirs, outcome)
    return outcome
