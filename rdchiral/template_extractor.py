import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from numpy.random import shuffle
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdmolfiles
from rdkit.Chem.rdchem import ChiralType

from rdchiral.utils import atoms_are_different

USE_STEREOCHEMISTRY = True
MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS = 5
INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS = True


_SPECIAL_GROUP_TEMPLATES: List[Tuple[List[int], Chem.Mol]] = []


def _initialize_special_group_templates() -> None:
    """Initialize pre-compiled SMARTS patterns for special groups.
    Called once at module import time."""
    global _SPECIAL_GROUP_TEMPLATES

    raw_templates: List[Tuple[List[int], str]] = [
        (
            list(range(3)),
            "[OH0,SH0]=C[O,Cl,I,Br,F]",
        ),  # carboxylic acid / halogen
        (
            list(range(3)),
            "[OH0,SH0]=CN",
        ),  # amide/sulfamide
        (
            list(range(4)),
            "S(O)(O)[Cl]",
        ),  # sulfonyl chloride
        (
            list(range(3)),
            "B(O)O",
        ),  # boronic acid/ester
        ([0], "[Si](C)(C)C"),  # trialkyl silane
        ([0], "[Si](OC)(OC)(OC)"),  # trialkoxy silane, default to methyl
        (
            list(range(3)),
            "[N;H0;$(N-[#6]);D2]-,=[N;D2]-,=[N;D1]",
        ),  # azide
        (
            list(range(8)),
            "O=C1N([Br,I,F,Cl])C(=O)CC1",
        ),  # NBS brominating agent
        (
            list(range(11)),
            "Cc1ccc(S(=O)(=O)O)cc1",
        ),  # Tosyl
        ([7], "CC(C)(C)OC(=O)[N]"),  # N(boc)
        ([4], "[CH3][CH0]([CH3])([CH3])O"),  #
        (
            list(range(2)),
            "[C,N]=[C,N]",
        ),  # alkene/imine
        (
            list(range(2)),
            "[C,N]#[C,N]",
        ),  # alkyne/nitrile
        (
            [2],
            "C=C-[*]",
        ),  # adj to alkene
        (
            [2],
            "C#C-[*]",
        ),  # adj to alkyne
        (
            [2],
            "O=C-[*]",
        ),  # adj to carbonyl
        ([3], "O=C([CH3])-[*]"),  # adj to methyl ketone
        (
            [3],
            "O=C([O,N])-[*]",
        ),  # adj to carboxylic acid/amide/ester
        (
            list(range(4)),
            "ClS(Cl)=O",
        ),  # thionyl chloride
        (
            list(range(2)),
            "[Mg,Li,Zn,Sn][Br,Cl,I,F]",
        ),  # grinard/metal (non-disassociated)
        (
            list(range(3)),
            "S(O)(O)",
        ),  # SO2 group
        (
            list(range(2)),
            "N~N",
        ),  # diazo
        (
            [1],
            "[!#6;R]@[#6;R]",
        ),  # adjacency to heteroatom in ring
        (
            [2],
            "[a!c]:a:a",
        ),  # two-steps away from heteroatom in aromatic ring
        # ((1,), 'c(-,=[*]):c([Cl,I,Br,F])',), # ortho to halogen on ring - too specific?
        # ((1,), 'c(-,=[*]):c:c([Cl,I,Br,F])',), # meta to halogen on ring - too specific?
        ([0], "[B,C](F)(F)F"),  # CF3, BF3 should have the F3 included
    ]

    # Stereo-specific ones (where we will need to include neighbors)
    # Tetrahedral centers should already be okay...
    raw_templates += [
        (
            [1, 2],
            "[*]/[CH]=[CH]/[*]",
        ),  # trans with two hydrogens
        (
            [1, 2],
            "[*]/[CH]=[CH]\[*]",
        ),  # cis with two hydrogens
        (
            [1, 2],
            "[*]/[CH]=[CH0]([*])\[*]",
        ),  # trans with one hydrogens
        (
            [1, 2],
            "[*]/[D3;H1]=[!D1]",
        ),  # specified on one end, can be N or C
    ]

    _SPECIAL_GROUP_TEMPLATES = [
        (add_if_match, rdmolfiles.MolFromSmarts(template))
        for add_if_match, template in raw_templates
    ]


_initialize_special_group_templates()


def mols_from_smiles_list(all_smiles: List[str]) -> List[Chem.Mol]:
    """Given a list of smiles strings, this function creates rdkit
    molecules"""
    mols: List[Chem.Mol] = []
    for smiles in all_smiles:
        if not smiles:
            continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols


def replace_deuterated(smi: str) -> str:
    return re.sub("\[2H\]", r"[H]", smi)


def clear_mapnum(mol: Chem.Mol) -> Chem.Mol:
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def get_tagged_atoms_from_mols(
    mols: List[Chem.Mol],
) -> Tuple[List[Chem.Atom], List[str]]:
    """Takes a list of RDKit molecules and returns total list of
    atoms and their tags"""
    atoms: List[Chem.Atom] = []
    atom_tags: List[str] = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol: Chem.Mol) -> Tuple[List[Chem.Atom], List[str]]:
    """Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers"""
    atoms: List[Chem.Atom] = []
    atom_tags: List[str] = []
    for atom in mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num:
            atoms.append(atom)
            atom_tags.append(str(atom_map_num))
    return atoms, atom_tags


def get_tetrahedral_atoms(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> List[Tuple[str, Chem.Atom, Chem.Atom]]:
    tetrahedral_atoms: List[Tuple[str, Chem.Atom, Chem.Atom]] = []

    reactant_atom_tags: Dict[str, Chem.Atom] = {}
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            atom_map_num = ar.GetAtomMapNum()
            if not atom_map_num:
                continue
            if ar.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                continue
            atom_tag = str(atom_map_num)
            reactant_atom_tags[atom_tag] = ar

    product_atom_tags: Dict[str, Chem.Atom] = {}
    for product in products:
        for ap in product.GetAtoms():
            atom_map_num = ap.GetAtomMapNum()
            if not atom_map_num:
                continue
            if ap.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                continue
            atom_tag = str(atom_map_num)
            product_atom_tags[atom_tag] = ap

    for atom_tag, ar in reactant_atom_tags.items():
        ap = product_atom_tags.get(atom_tag)
        if ap is not None:
            tetrahedral_atoms.append((atom_tag, ar, ap))
    return tetrahedral_atoms


def get_frag_around_tetrahedral_center(mol: Chem.Mol, idx: int) -> str:
    """Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes"""
    ids_to_include: List[int] = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols: List[str] = [
        "[{}{}]".format(a.GetIsotope(), a.GetSymbol())
        if a.GetIsotope() != 0
        else "[#{}]".format(a.GetAtomicNum())
        for a in mol.GetAtoms()
    ]
    return rdmolfiles.MolFragmentToSmiles(
        mol,
        ids_to_include,
        isomericSmiles=True,
        atomSymbols=symbols,
        allBondsExplicit=True,
        allHsExplicit=True,
    )


def check_tetrahedral_centers_equivalent(atom1: Chem.Atom, atom2: Chem.Atom) -> bool:
    """Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped"""
    atom1_frag = get_frag_around_tetrahedral_center(
        atom1.GetOwningMol(), atom1.GetIdx()
    )
    atom2_idx = atom2.GetIdx()
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    if atom1_neighborhood is None:
        return False
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(
        atom1_neighborhood, useChirality=True
    ):
        if atom2_idx in matched_ids:
            return True
    return False


def clear_isotope(mol: Chem.Mol) -> None:
    for a in mol.GetAtoms():
        a.SetIsotope(0)


def get_changed_atoms(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> Tuple[List[Chem.Atom], List[str], int]:
    """Looks at mapped atoms in a reaction and determines which ones changed"""

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)

    # Find differences
    changed_atoms: List[Chem.Atom] = []  # actual reactant atom species
    changed_atom_tags: List[str] = []  # atom map numbers of those atoms

    prod_tag_counts = Counter(prod_atom_tags)
    prod_atom_by_tag: Dict[str, Chem.Atom] = {}
    for atom, tag in zip(prod_atoms, prod_atom_tags):
        if tag not in prod_atom_by_tag:
            prod_atom_by_tag[tag] = atom

    reac_atom_by_tag: Dict[str, Chem.Atom] = {}
    for atom, tag in zip(reac_atoms, reac_atom_tags):
        if tag not in reac_atom_by_tag:
            reac_atom_by_tag[tag] = atom

    changed_tag_set = set()

    # Reactant atoms that do not appear in product (tagged leaving groups)
    prod_tag_set = set(prod_atom_tags)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag in changed_tag_set:
            continue
        if reac_tag not in prod_tag_set:
            changed_atoms.append(reac_atoms[j])
            changed_atom_tags.append(reac_tag)
            changed_tag_set.add(reac_tag)

    # Product atoms that are different from reactant atom equivalent
    for tag, reac_atom in reac_atom_by_tag.items():
        prod_atom = prod_atom_by_tag.get(tag)
        if prod_atom is None:
            continue
        if tag in changed_tag_set:
            continue
        if prod_tag_counts[tag] > 1 or atoms_are_different(prod_atom, reac_atom):
            changed_atoms.append(reac_atom)
            changed_atom_tags.append(tag)
            changed_tag_set.add(tag)

    # Atoms that change CHIRALITY (just tetrahedral for now...)
    tetra_atoms = get_tetrahedral_atoms(reactants, products)

    for atom_tag, ar, ap in tetra_atoms:
        if atom_tag in changed_atom_tags:
            continue
        else:
            unchanged = check_tetrahedral_centers_equivalent(
                ar, ap
            ) and ChiralType.CHI_UNSPECIFIED not in [
                ar.GetChiralTag(),
                ap.GetChiralTag(),
            ]
            if unchanged:
                continue
            else:
                # Make sure chiral change is next to the reaction center and not
                # a random specifidation (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    neighbor_map_num = str(neighbor.GetAtomMapNum())
                    if neighbor_map_num in changed_atom_tags:
                        tetra_adj_to_rxn = True
                        break
                if tetra_adj_to_rxn:
                    changed_atom_tags.append(str(atom_tag))
                    changed_atoms.append(ar)

    return changed_atoms, changed_atom_tags, err


def get_special_groups(mol: Chem.Mol) -> List[Tuple[List[int], List[int]]]:
    """Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should
    be included in a fragment all together. This should only be done for the
    reactants, otherwise the products might end up with mapping mismatches

    We draw a distinction between atoms in groups that trigger that whole
    group to be included, and "unimportant" atoms in the groups that will not
    be included if another atom matches."""

    # Build list
    groups: List[Tuple[List[int], List[int]]] = []
    for add_if_match, template_mol in _SPECIAL_GROUP_TEMPLATES:
        if template_mol is None:
            continue
        matches = mol.GetSubstructMatches(template_mol, useChirality=True)
        for match in matches:
            add_if: List[int] = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, list(match)))
    return groups


def expand_atoms_to_use(
    mol: Chem.Mol,
    atoms_to_use: List[int],
    groups: Optional[List[Any]] = None,
    symbol_replacements: Optional[List[Tuple[int, str]]] = None,
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Given an RDKit molecule and a list of AtomIdX which should be included
    in the reaction, this function expands the list of AtomIdXs to include one
    nearest neighbor with special consideration of (a) unimportant neighbors and
    (b) important functional groupings"""

    if groups is None:
        groups = []

    if symbol_replacements is None:
        symbol_replacements = []

    # Copy
    new_atoms_to_use = atoms_to_use[:]

    # Look for all atoms in the current list of atoms to use
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_use:
            continue
        # Ensure membership of changed atom is checked against group
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                for idx in group[1]:
                    if idx not in atoms_to_use:
                        new_atoms_to_use.append(idx)
                        symbol_replacements.append(
                            (idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(idx)))
                        )
        # Look for all nearest neighbors of the currently-included atoms
        for neighbor in atom.GetNeighbors():
            # Evaluate nearest neighbor atom to determine what should be included
            new_atoms_to_use, symbol_replacements = expand_atoms_to_use_atom(
                mol,
                new_atoms_to_use,
                neighbor.GetIdx(),
                groups=groups,
                symbol_replacements=symbol_replacements,
            )

    return new_atoms_to_use, symbol_replacements


def expand_atoms_to_use_atom(
    mol: Chem.Mol,
    atoms_to_use: List[int],
    atom_idx: int,
    groups: Optional[List[Tuple[List[int], List[int]]]] = None,
    symbol_replacements: Optional[List[Tuple[int, str]]] = None,
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Given an RDKit molecule and a list of AtomIdx which should be included
    in the reaction, this function extends the list of atoms_to_use by considering
    a candidate atom extension, atom_idx"""

    if not groups:
        groups = []

    if not symbol_replacements:
        symbol_replacements = []

    # See if this atom belongs to any special groups (highest priority)
    found_in_group = False
    for group in groups:  # first index is atom IDs for match, second is what to include
        if int(atom_idx) in group[0]:  # int correction
            # Add the whole list, redundancies don't matter
            # *but* still call convert_atom_to_wildcard!
            for idx in group[1]:
                if idx not in atoms_to_use:
                    atoms_to_use.append(idx)
                    symbol_replacements.append(
                        (idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(idx)))
                    )
            found_in_group = True
    if found_in_group:
        return atoms_to_use, symbol_replacements

    # How do we add an atom that wasn't in an identified important functional group?
    # Develop generalized SMARTS symbol

    # Skip current candidate atom if it is already included
    if atom_idx in atoms_to_use:
        return atoms_to_use, symbol_replacements

    # Include this atom
    atoms_to_use.append(atom_idx)

    # Look for suitable SMARTS replacement
    symbol_replacements.append(
        (atom_idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(atom_idx)))
    )

    return atoms_to_use, symbol_replacements


def convert_atom_to_wildcard(atom: Chem.Atom) -> str:
    """This function takes an RDKit atom and turns it into a wildcard
    using heuristic generalization rules. This function should be used
    when candidate atoms are used to extend the reaction core for higher
    generalizability"""

    # Is this a terminal atom? We can tell if the degree is one
    if atom.GetDegree() == 1:
        symbol = "[" + atom.GetSymbol() + ";D1;H{}".format(atom.GetTotalNumHs())
        if atom.GetFormalCharge() != 0:
            charges = re.search("([-+]+[1-9]?)", atom.GetSmarts())
            if charges:
                symbol = symbol.replace(";D1", ";{};D1".format(charges.group()))

    else:
        # Initialize
        symbol = "["

        # Add atom primitive - atomic num and aromaticity (don't use COMPLETE wildcards)
        if atom.GetAtomicNum() != 6:
            symbol += "#{};".format(atom.GetAtomicNum())
            if atom.GetIsAromatic():
                symbol += "a;"
        elif atom.GetIsAromatic():
            symbol += "c;"
        else:
            symbol += "C;"

        # Charge is important
        if atom.GetFormalCharge() != 0:
            charges = re.search("([-+]+[1-9]?)", atom.GetSmarts())
            if charges:
                symbol += charges.group() + ";"

        # Strip extra semicolon
        if symbol[-1] == ";":
            symbol = symbol[:-1]

    # Close with label or with bracket
    label = re.search("\:[0-9]+\]", atom.GetSmarts())
    if label:
        symbol += label.group()
    else:
        symbol += "]"

    return symbol


def reassign_atom_mapping(transform: str) -> str:
    """This function takes an atom-mapped reaction SMILES and reassigns
    the atom-mapping labels (numbers) from left to right, once
    that transform has been canonicalized."""

    all_labels: List[str] = re.findall("\:([0-9]+)\]", transform)

    # Define list of replacements which matches all_labels *IN ORDER*
    replacements: List[str] = []
    replacement_dict: Dict[str, str] = {}
    counter = 1
    for label in all_labels:  # keep in order! this is important
        if label not in replacement_dict:
            replacement_dict[label] = str(counter)
            counter += 1
        replacements.append(replacement_dict[label])

    # Perform replacements in order
    transform_newmaps = re.sub(
        "\:[0-9]+\]", lambda match: ":" + replacements.pop(0) + "]", transform
    )

    return transform_newmaps


def get_strict_smarts_for_atom(atom: Chem.Atom) -> str:
    """
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    """

    symbol = atom.GetSmarts()
    if atom.GetSymbol() == "H":
        symbol = "[#1]"

    if "[" not in symbol:
        symbol = "[" + symbol + "]"

    # Explicit stereochemistry - *before* H
    if USE_STEREOCHEMISTRY:
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            if "@" not in symbol:
                # Be explicit when there is a tetrahedral chiral tag
                if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    tag = "@"
                elif atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    tag = "@@"
                if ":" in symbol:
                    symbol = symbol.replace(":", ";{}:".format(tag))
                else:
                    symbol = symbol.replace("]", ";{}]".format(tag))

    if "H" not in symbol:
        H_symbol = "H{}".format(atom.GetTotalNumHs())
        # Explicit number of hydrogens: include "H0" when no hydrogens present
        if ":" in symbol:  # stick H0 before label
            symbol = symbol.replace(":", ";{}:".format(H_symbol))
        else:
            symbol = symbol.replace("]", ";{}]".format(H_symbol))

    # Explicit degree
    if ":" in symbol:
        symbol = symbol.replace(":", ";D{}:".format(atom.GetDegree()))
    else:
        symbol = symbol.replace("]", ";D{}]".format(atom.GetDegree()))

    # Explicit formal charge
    if "+" not in symbol and "-" not in symbol:
        charge = atom.GetFormalCharge()
        charge_symbol = "+" if (charge >= 0) else "-"
        charge_symbol += "{}".format(abs(charge))
        if ":" in symbol:
            symbol = symbol.replace(":", ";{}:".format(charge_symbol))
        else:
            symbol = symbol.replace("]", ";{}]".format(charge_symbol))

    return symbol


def expand_changed_atom_tags(
    changed_atom_tags: List[str], reactant_fragments: str
) -> List[str]:
    """Given a list of changed atom tags (numbers as strings) and a string consisting
    of the reactant_fragments to include in the reaction transform, this function
    adds any tagged atoms found in the reactant side of the template to the
    changed_atom_tags list so that those tagged atoms are included in the products"""

    expansion = []
    atom_tags_in_reactant_fragments: List[str] = re.findall(
        "\:([0-9]+)\]", reactant_fragments
    )
    for atom_tag in atom_tags_in_reactant_fragments:
        if atom_tag not in changed_atom_tags:
            expansion.append(atom_tag)
    return expansion


def get_fragments_for_changed_atoms(
    mols: List[Chem.Mol],
    changed_atom_tags: List[str],
    radius: int = 0,
    category: str = "reactants",
    expansion: Optional[List[str]] = None,
    no_special_groups: bool = False,
) -> Tuple[str, bool, bool]:
    """Given a list of RDKit mols and a list of changed atom tags, this function
    computes the SMILES string of molecular fragments using MolFragmentToSmiles
    for all changed fragments.

    expansion: atoms added during reactant expansion that should be included and
               generalized in product fragment
    """
    if expansion is None:
        expansion = []

    fragments = ""
    mols_changed = []
    for mol in mols:
        # Initialize list of replacement symbols (updated during expansion)
        symbol_replacements = []

        # Are we looking for special reactive groups? (reactants only)
        if category == "reactants" and not no_special_groups:
            groups = get_special_groups(mol)
        else:
            groups = []

        # Build list of atoms to use
        atoms_to_use = []
        for atom in mol.GetAtoms():
            # Check self (only tagged atoms)
            if ":" in atom.GetSmarts():
                if atom.GetSmarts().split(":")[1][:-1] in changed_atom_tags:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    if symbol != atom.GetSmarts():
                        symbol_replacements.append((atom.GetIdx(), symbol))
                    continue

        # Fully define leaving groups and this molecule participates?
        if INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS and len(atoms_to_use) > 0:
            if category == "reactants":
                for atom in mol.GetAtoms():
                    if not atom.GetAtomMapNum():
                        atoms_to_use.append(atom.GetIdx())

        # Check neighbors (any atom)
        for k in range(radius):
            atoms_to_use, symbol_replacements = expand_atoms_to_use(
                mol,
                atoms_to_use,
                groups=groups,
                symbol_replacements=symbol_replacements,
            )

        if category == "products":
            # Add extra labels to include (for products only)
            if expansion:
                for atom in mol.GetAtoms():
                    if ":" not in atom.GetSmarts():
                        continue
                    label = atom.GetSmarts().split(":")[1][:-1]
                    if label in expansion and label not in changed_atom_tags:
                        atoms_to_use.append(atom.GetIdx())
                        # Make the expansion a wildcard
                        symbol_replacements.append(
                            (atom.GetIdx(), convert_atom_to_wildcard(atom))
                        )

            # Make sure unmapped atoms are included (from products)
            for atom in mol.GetAtoms():
                if not atom.GetAtomMapNum():
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    symbol_replacements.append((atom.GetIdx(), symbol))

        if not atoms_to_use:
            continue

        # Define new symbols based on symbol_replacements
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for i, symbol in symbol_replacements:
            symbols[i] = symbol

        mol_smi_with_maps = Chem.MolToSmiles(mol, True)
        mol_from_smi_with_maps = Chem.MolFromSmiles(mol_smi_with_maps)
        if mol_from_smi_with_maps is None:
            raise ValueError("could not parse molecule SMILES")
        mols_changed.append(
            Chem.MolToSmiles(clear_mapnum(mol_from_smi_with_maps), True)
        )

        # Keep flipping stereocenters until we are happy...
        # this is a sloppy fix during extraction to achieve consistency

        tetra_consistent = False
        num_tetra_flips = 0
        mol_without_map_nums = Chem.Mol(mol)
        for a in mol_without_map_nums.GetAtoms():
            a.SetAtomMapNum(0)
        while not tetra_consistent and num_tetra_flips < 100:
            mol_copy = Chem.Mol(mol_without_map_nums)
            this_fragment = rdmolfiles.MolFragmentToSmiles(
                mol_copy,
                atoms_to_use,
                atomSymbols=symbols,
                allHsExplicit=True,
                isomericSmiles=USE_STEREOCHEMISTRY,
                allBondsExplicit=True,
            )

            # Figure out what atom maps are tetrahedral centers
            # Set isotopes to make sure we're getting the *exact* match we want
            this_fragment_mol = rdmolfiles.MolFromSmarts(this_fragment)
            if this_fragment_mol is None:
                raise ValueError("could not parse fragment SMARTS")
            tetra_map_nums = []
            for atom in this_fragment_mol.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num:
                    atom.SetIsotope(atom_map_num)
                    if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                        tetra_map_nums.append(atom_map_num)

            # If there are no mapped tetrahedral stereocenters in the fragment,
            # there is nothing to validate/flip via chirality-aware matching.
            if not tetra_map_nums:
                tetra_consistent = True
                break
            map_to_id = {}
            for atom in mol.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num:
                    atom.SetIsotope(atom_map_num)
                    map_to_id[atom_map_num] = atom.GetIdx()
                else:
                    atom.SetIsotope(0)

            # Look for matches
            tetra_consistent = True
            all_matched_ids = []

            # skip substructure matching if there are a lot of fragments
            # this can help prevent GetSubstructMatches from hanging
            frag_smi = Chem.MolToSmiles(this_fragment_mol)
            if frag_smi.count(".") > 5:
                break

            for matched_ids in mol.GetSubstructMatches(
                this_fragment_mol, useChirality=True
            ):
                all_matched_ids.extend(matched_ids)
            shuffle(tetra_map_nums)
            for tetra_map_num in tetra_map_nums:
                if map_to_id[tetra_map_num] not in all_matched_ids:
                    tetra_consistent = False
                    prevsymbol = symbols[map_to_id[tetra_map_num]]
                    if "@@" in prevsymbol:
                        symbol = prevsymbol.replace("@@", "@")
                    elif "@" in prevsymbol:
                        symbol = prevsymbol.replace("@", "@@")
                    else:
                        raise ValueError(
                            "Need to modify symbol of tetra atom without @ or @@??"
                        )
                    symbols[map_to_id[tetra_map_num]] = symbol
                    num_tetra_flips += 1
                    # IMPORTANT: only flip one at a time
                    break

        if not tetra_consistent:
            raise ValueError(
                "Could not find consistent tetrahedral mapping, {} centers".format(
                    len(tetra_map_nums)
                )
            )

        fragments += "(" + this_fragment + ")."

    # auxiliary template information: is this an intramolecular reaction or dimerization?
    intra_only = 1 == len(mols_changed)
    dimer_only = (1 == len(set(mols_changed))) and (len(mols_changed) == 2)

    return fragments[:-1], intra_only, dimer_only


def canonicalize_transform(transform: str) -> str:
    """This function takes an atom-mapped SMARTS transform and
    converts it to a canonical form by, if nececssary, rearranging
    the order of reactant and product templates and reassigning
    atom maps."""

    transform_reordered = ">>".join(
        [canonicalize_template(x) for x in transform.split(">>")]
    )
    return reassign_atom_mapping(transform_reordered)


def canonicalize_template(template: str) -> str:
    """This function takes one-half of a template SMARTS string
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping."""

    # Strip labels to get sort orders
    template_nolabels = re.sub("\:[0-9]+\]", "]", template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(").(")
    template_mols = template[1:-1].split(").(")

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split(".")
        mol_frags = template_mols[i].split(".")

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [
            j[0] for j in sorted(enumerate(nolabel_mol_frags), key=lambda x: x[1])
        ]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = ".".join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i] = ".".join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [
        j[0] for j in sorted(enumerate(template_nolabels_mols), key=lambda x: x[1])
    ]

    # Apply sorting and merge list back into overall transform
    template = "(" + ").(".join([template_mols[i] for i in sortorder]) + ")"

    return template


def extract_from_reaction(
    reaction: Dict[str, Any], no_special_groups: bool = False, radius: int = 1
) -> Dict[str, Any]:
    reactants = mols_from_smiles_list(
        replace_deuterated(reaction["reactants"]).split(".")
    )
    products = mols_from_smiles_list(
        replace_deuterated(reaction["products"]).split(".")
    )

    # if rdkit cant understand molecule, return
    if None in reactants:
        return {"reaction_id": reaction["_id"]}
    if None in products:
        return {"reaction_id": reaction["_id"]}

    are_unmapped_product_atoms = False
    num_unmapped_product_atoms = 0
    unmapped_ids = []
    seen_atom_map_nums = set()
    for product in products:
        prod_atoms = product.GetAtoms()
        num_mapped_atoms = 0
        for atom in prod_atoms:
            map_num = atom.GetAtomMapNum()
            if map_num:
                seen_atom_map_nums.add(map_num)
                num_mapped_atoms += 1
            else:
                num_unmapped_product_atoms += 1
                unmapped_ids.append(atom.GetIdx())
                if num_unmapped_product_atoms > MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                    # Skip this example - too many unmapped product atoms!
                    return {"reaction_id": reaction["_id"]}
        if num_mapped_atoms < len(prod_atoms):
            are_unmapped_product_atoms = True

    reactants_in_reaction = []
    for reactant in reactants:
        for atom in reactant.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num and map_num in seen_atom_map_nums:
                reactants_in_reaction.append(reactant)
                break
    _ = [reactant for reactant in reactants if reactant not in reactants_in_reaction]
    reactants = reactants_in_reaction
    if not reactants:
        return {"reaction_id": reaction["_id"]}

    # try to sanitize molecules
    try:
        for i in range(len(reactants)):
            reactants[i] = Chem.RemoveHs(reactants[i])  # *might* not be safe
        for i in range(len(products)):
            products[i] = Chem.RemoveHs(products[i])  # *might* not be safe
        for mol in reactants + products:
            mol = Chem.SanitizeMol(mol)  # redundant w/ RemoveHs
        for mol in reactants + products:
            mol.UpdatePropertyCache()
    except Exception:
        # can't sanitize -> skip
        return {"reaction_id": reaction["_id"]}

    if None in reactants + products:
        return {"reaction_id": reaction["_id"]}

    extra_reactant_fragment = ""
    if are_unmapped_product_atoms:  # add fragment to template
        for product in products:
            prod_atoms = product.GetAtoms()
            # Define new atom symbols for fragment with atom maps, generalizing fully
            atom_symbols = ["[{}]".format(a.GetSymbol()) for a in prod_atoms]
            # And bond symbols...
            bond_symbols = ["~" for _ in product.GetBonds()]
            if unmapped_ids:
                extra_reactant_fragment += (
                    rdmolfiles.MolFragmentToSmiles(
                        product,
                        unmapped_ids,
                        allHsExplicit=False,
                        isomericSmiles=USE_STEREOCHEMISTRY,
                        atomSymbols=atom_symbols,
                        bondSymbols=bond_symbols,
                    )
                    + "."
                )
        if extra_reactant_fragment:
            extra_reactant_fragment = extra_reactant_fragment[:-1]

        # Consolidate repeated fragments (stoichometry)
        extra_reactant_fragment = ".".join(
            sorted(list(set(extra_reactant_fragment.split("."))))
        )

    # Calculate changed atoms
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    if err:
        return {"reaction_id": reaction["_id"]}
    if not changed_atom_tags:
        return {"reaction_id": reaction["_id"]}

    try:
        # Get fragments for reactants
        reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(
            reactants,
            changed_atom_tags,
            radius=radius,
            expansion=[],
            category="reactants",
            no_special_groups=no_special_groups,
        )
        # Get fragments for products
        # (WITHOUT matching groups but WITH the addition of reactant fragments)
        product_fragments, _, _ = get_fragments_for_changed_atoms(
            products,
            changed_atom_tags,
            radius=0,
            expansion=expand_changed_atom_tags(changed_atom_tags, reactant_fragments),
            category="products",
        )
    except ValueError:
        return {"reaction_id": reaction["_id"]}

    # Put together and canonicalize (as best as possible)
    rxn_string = "{}>>{}".format(reactant_fragments, product_fragments)
    rxn_canonical = canonicalize_transform(rxn_string)
    # Change from inter-molecular to intra-molecular
    rxn_canonical_split = rxn_canonical.split(">>")
    rxn_canonical = (
        rxn_canonical_split[0][1:-1].replace(").(", ".")
        + ">>"
        + rxn_canonical_split[1][1:-1].replace(").(", ".")
    )

    reactants_string = rxn_canonical.split(">>")[0]
    products_string = rxn_canonical.split(">>")[1]

    retro_canonical = products_string + ">>" + reactants_string

    # Load into RDKit
    rxn: Optional[rdChemReactions.ChemicalReaction] = (
        rdChemReactions.ReactionFromSmarts(retro_canonical)
    )
    if rxn is None:
        return {"reaction_id": reaction["_id"]}
    if rxn.Validate()[1] != 0:
        return {"reaction_id": reaction["_id"]}

    template = {
        "products": products_string,
        "reactants": reactants_string,
        "reaction_smarts": retro_canonical,
        "intra_only": intra_only,
        "dimer_only": dimer_only,
        "reaction_id": reaction["_id"],
        "necessary_reagent": extra_reactant_fragment,
    }

    return template
