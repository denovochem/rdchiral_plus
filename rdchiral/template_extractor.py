import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdmolfiles
from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType

from rdchiral.chiral import atom_chirality_matches
from rdchiral.utils import atoms_are_different


class ExtractedTemplate(TypedDict):
    products: str
    reactants: str
    spectators: str
    reaction_smarts: str
    separated_reaction_smarts: List[str]
    intra_only: bool
    dimer_only: bool
    reaction_id: str | int | None
    necessary_reagent: str


DEFAULT_EXTRACTED_TEMPLATE: ExtractedTemplate = {
    "products": "",
    "reactants": "",
    "spectators": "",
    "reaction_smarts": "",
    "separated_reaction_smarts": [],
    "intra_only": False,
    "dimer_only": False,
    "reaction_id": "",
    "necessary_reagent": "",
}


class rdChiralTemplateExtractInput(TypedDict):
    reactants: str
    products: str
    _id: str | int | None


_SPECIAL_GROUP_TEMPLATES: List[Tuple[List[int], Chem.Mol]] = []


def _initialize_special_group_templates() -> None:
    """
    Initialize pre-compiled SMARTS patterns for special chemical groups.

    Compiles SMARTS strings into RDKit molecule objects for use in identifying
    special functional groups and structural patterns during template extraction.
    The compiled templates are stored in the global _SPECIAL_GROUP_TEMPLATES
    variable.

    Each template consists of a tuple (add_if_match, pattern) where:
    - add_if_match (List[int]): Atom indices that trigger inclusion of the group.
    - pattern (Chem.Mol): A compiled SMARTS pattern for matching.

    Covered groups include common functional groups (carboxylic acids, amides,
    boronic acids, azides, etc.), metals (Grignard reagents), stereospecific
    patterns (alkenes with defined stereochemistry), and structural patterns
    (adjacency to carbonyls, aromatic heteroatoms).

    Returns:
        None

    Note:
        This function is called once at module import time. It modifies the
        global _SPECIAL_GROUP_TEMPLATES variable in-place. If SMARTS compilation
        fails, the corresponding template is silently excluded (template_mol will
        be None).
    """
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
        ([4], "[CH3][CH0]([CH3])([CH3])O"),
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
            r"[*]/[CH]=[CH]\[*]",
        ),  # cis with two hydrogens
        (
            [1, 2],
            r"[*]/[CH]=[CH0]([*])\[*]",
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


def _invert_smarts_chirality_for_match(
    match: re.Match, unmapped_only: bool = True
) -> str:
    """
    Invert the tetrahedral chirality symbol (@ or @@) in a SMARTS atom token.

    This is a helper function for `invert_chirality_around_unmapped_ring_closure`.
    Given a regex match containing a tetrahedral chirality token (e.g., [C@H] or [C@@H]),
    returns the token with inverted chirality ([C@@H] or [C@H] respectively).

    Args:
        match (re.Match): A regex match object matching the chirality portion of a SMARTS atom token.
        unmapped_only (bool): If True, only invert chirality for unmapped atoms (those without a colon
            in the token, which indicates atom mapping). Defaults to True.

    Returns:
        str: The (possibly inverted) tetrahedral token as a string.

    Note:
        When unmapped_only is True and the token contains ':' (indicating an atom map number),
        the original token is returned unchanged. This prevents modifying mapped atoms which
        should retain their original chirality in reaction templates.
    """
    # only want to change unmapped token:
    tetrahedral_token = match.group(1)
    if ":" in tetrahedral_token and unmapped_only:
        return str(tetrahedral_token)

    if "@@" in tetrahedral_token:
        return str(tetrahedral_token.replace("@@", "@"))
    elif "@" in tetrahedral_token:
        return str(tetrahedral_token.replace("@", "@@"))
    else:
        return str(tetrahedral_token)


def invert_chirality_around_unmapped_ring_closure(smarts: str) -> str:
    """
    Invert tetrahedral chirality for unmapped atoms preceding ring closure tokens.

    This function processes a SMARTS string and flips the chirality symbol (@ <-> @@)
    for carbon atoms that appear immediately before a ring closure digit (1-9). Only
    unmapped atoms (those without atom map numbers) are modified.

    Args:
        smarts (str): A SMARTS string representing a molecular pattern.

    Returns:
        str: The modified SMARTS string with inverted chirality for matching atoms.

    Note:
        This is used in reaction template extraction to handle chirality invariants
        when atoms participate in ring closures. The regex pattern matches tetrahedral
        carbon tokens ([C@...] or [C@@...]) that are followed by a ring closure digit.
    """
    return re.sub(
        r"(\[C@+[^\]]*\])(?=.?[1-9])", _invert_smarts_chirality_for_match, smarts
    )


def mols_from_smiles_list(all_smiles: List[str]) -> List[Chem.Mol]:
    """
    Convert a list of SMILES strings to RDKit molecules.

    Empty strings in the input list are skipped. Invalid SMILES strings
    will result in None values in the returned list.

    Args:
        all_smiles (List[str]): A list of SMILES strings to convert.

    Returns:
        List[Chem.Mol]: A list of RDKit molecules. None values may be
            present for invalid SMILES strings.
    """
    mols: List[Chem.Mol] = []
    for smiles in all_smiles:
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
    return mols


def replace_deuterated(smi: str) -> str:
    """
    Replace deuterium atoms with regular hydrogen atoms in a SMILES string.

    Args:
        smi (str): The SMILES string potentially containing deuterium atoms.

    Returns:
        str: The SMILES string with all [2H] deuterium atoms replaced by [H].
    """
    return re.sub(r"\[2H\]", r"[H]", smi)


def clear_mapnum(mol: Chem.Mol) -> Chem.Mol:
    """
    Clear all atom map numbers from a molecule by setting them to 0.

    Args:
        mol (Chem.Mol): The RDKit molecule whose atom map numbers will be cleared.

    Returns:
        Chem.Mol: The same molecule with all atom map numbers set to 0.
    """
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def get_tagged_atoms_from_mols(
    mols: List[Chem.Mol],
) -> Tuple[List[Chem.Atom], List[int]]:
    """
    Collect all tagged atoms and their map numbers from a list of RDKit molecules.

    A "tagged" atom is one with a non-zero atom map number. This function iterates
    over multiple molecules and aggregates all tagged atoms and their corresponding
    map numbers into flat lists.

    Args:
        mols (List[Chem.Mol]): A list of RDKit molecules to search for tagged atoms.

    Returns:
        Tuple[List[Chem.Atom], List[int]]:
            - First element: List of tagged atoms (atoms with non-zero map numbers).
            - Second element: List of atom map numbers corresponding to each atom
              in the first list.

    Note:
        Atoms and their tags maintain corresponding indices in the two returned lists.
        The order of atoms follows the iteration order of molecules in the input list
        and the atom order within each molecule.
    """
    atoms: List[Chem.Atom] = []
    atom_tags: List[int] = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms.extend(new_atoms)
        atom_tags.extend(new_atom_tags)
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol: Chem.Mol) -> Tuple[List[Chem.Atom], List[int]]:
    """
    Extract atoms with non-zero atom map numbers from an RDKit molecule.

    Args:
        mol (Chem.Mol): The molecule to extract tagged atoms from.

    Returns:
        Tuple[List[Chem.Atom], List[int]]:
            - First list: Atoms that have non-zero atom map numbers.
            - Second list: Corresponding atom map numbers for each atom.

    Note:
        Atoms without atom map numbers (GetAtomMapNum() == 0) are excluded.
        The order of atoms in the returned lists follows the iteration order of the molecule.
    """
    atoms: List[Chem.Atom] = []
    atom_tags: List[int] = []
    for atom in mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num:
            atoms.append(atom)
            atom_tags.append(atom_map_num)
    return atoms, atom_tags


def get_tetrahedral_atoms(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> List[Tuple[int, Chem.Atom, Chem.Atom]]:
    """
    Identify tetrahedral (chiral) atoms that are mapped between reactants and products.

    This function finds atoms with specified chiral tags in both reactants and products
    that share the same atom map number. Atoms with any chiral tag other than
    CHI_UNSPECIFIED and a valid (non-zero) atom map number are considered.

    Args:
        reactants (List[Chem.Mol]): List of reactant molecules to search for chiral atoms.
        products (List[Chem.Mol]): List of product molecules to search for chiral atoms.

    Returns:
        List[Tuple[int, Chem.Atom, Chem.Atom]]: A list of tuples, where each tuple contains:
            - The atom map number (int) identifying the matched chiral center.
            - The corresponding chiral atom from the reactant (Chem.Atom).
            - The corresponding chiral atom from the product (Chem.Atom).
    """
    tetrahedral_atoms: List[Tuple[int, Chem.Atom, Chem.Atom]] = []

    reactant_atom_tags: Dict[int, Chem.Atom] = {}
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            atom_map_num = ar.GetAtomMapNum()
            if not atom_map_num:
                continue
            if ar.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                continue
            atom_tag = atom_map_num
            reactant_atom_tags[atom_tag] = ar

    product_atom_tags: Dict[int, Chem.Atom] = {}
    for product in products:
        for ap in product.GetAtoms():
            atom_map_num = ap.GetAtomMapNum()
            if not atom_map_num:
                continue
            if ap.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                continue
            atom_tag = atom_map_num
            product_atom_tags[atom_tag] = ap

    for atom_tag, ar in reactant_atom_tags.items():
        ap = product_atom_tags.get(atom_tag)
        if ap is not None:
            tetrahedral_atoms.append((atom_tag, ar, ap))

    return tetrahedral_atoms


def get_stereogenic_double_bonds(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> List[Tuple[int, int]]:
    """
    Identify double bonds with stereochemical differences between reactants and products.

    Compares double bond stereochemistry between corresponding atoms in reactants and
    products. A bond is considered stereogenic if stereochemistry is explicitly defined
    on either side of the reaction and differs between reactants and products.

    Args:
        reactants (List[Chem.Mol]): List of reactant molecules with atom map numbers.
        products (List[Chem.Mol]): List of product molecules with atom map numbers.

    Returns:
        List[Tuple[int, int]]: List of atom map number pairs identifying stereogenic
            double bonds. Each tuple contains two sorted atom map numbers.

    Note:
        Bonds are matched using atom map numbers. Only double bonds with atom map
        numbers on both atoms are considered. Bonds present in only one side of the
        reaction are included if they have defined stereochemistry.
    """
    # Build map of atom map numbers to bonds in reactants
    reactant_bonds: Dict[Tuple[int, int], BondStereo] = {}
    for reactant in reactants:
        for bond in reactant.GetBonds():
            if bond.GetBondType() != BondType.DOUBLE:
                continue

            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            if atom1_map and atom2_map:
                key = tuple(sorted([atom1_map, atom2_map]))
                reactant_bonds[key] = bond.GetStereo()

    # Build map for products
    product_bonds: Dict[Tuple[int, int], BondStereo] = {}
    for product in products:
        for bond in product.GetBonds():
            if bond.GetBondType() != BondType.DOUBLE:
                continue

            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            if atom1_map and atom2_map:
                key = tuple(sorted([atom1_map, atom2_map]))
                product_bonds[key] = bond.GetStereo()

    # Find bonds that have stereochemistry that changes or is specified
    stereogenic_bonds = []
    all_keys = set(reactant_bonds.keys()) | set(product_bonds.keys())

    for key in all_keys:
        r_stereo = reactant_bonds.get(key, BondStereo.STEREONONE)
        p_stereo = product_bonds.get(key, BondStereo.STEREONONE)

        # If stereo is specified in either side and they differ, or if specified on one side but not the other
        if (
            r_stereo != BondStereo.STEREONONE or p_stereo != BondStereo.STEREONONE
        ) and r_stereo != p_stereo:
            stereogenic_bonds.append(key)

    return stereogenic_bonds


def ensure_complete_stereo_double_bonds(
    mol: Chem.Mol,
    atoms_to_use: List[int],
    symbol_replacements: List[Tuple[int, str]],
    use_stereochemistry: bool = True,
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """
    Ensures that for any stereogenic double bond in the fragment, all necessary
    atoms (the bond atoms + one neighbor on each side) are included.

    When extracting a substructure fragment, stereogenic double bonds require
    all atoms of the bond plus at least one neighbor on each side to preserve
    stereochemical information. This function scans the atoms in the fragment,
    identifies any stereogenic double bonds, and ensures both bond atoms and
    at least one neighbor per bond atom are included in the fragment.

    Args:
        mol (Chem.Mol): The full molecule from which the fragment is extracted.
        atoms_to_use (List[int]): List of atom indices currently in the fragment.
        symbol_replacements (List[Tuple[int, str]]): List of (atom_idx, SMARTS)
            pairs for atoms that should use custom SMARTS patterns in the
            template instead of their default atomic representation.
        use_stereochemistry (bool): If False, returns inputs unchanged without
            processing stereochemistry. Defaults to True.

    Returns:
        Tuple[List[int], List[Tuple[int, str]]]:
            - Updated list of atom indices with any missing stereochem-required
              atoms appended.
            - Updated list of symbol replacements with entries for newly added
              atoms (bond atoms use strict SMARTS, neighbors use wildcards).
    """
    if not use_stereochemistry:
        return atoms_to_use, symbol_replacements
    # Use a set for efficient checking and a list to maintain order
    atoms_to_use_set = set(atoms_to_use)

    # Iterate through a copy of atoms_to_use to avoid issues with modification during loop
    for atom_idx in list(atoms_to_use_set):
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            if (
                bond.GetBondType() != BondType.DOUBLE
                or bond.GetStereo() == BondStereo.STEREONONE
            ):
                continue

            # This is a stereogenic double bond connected to our fragment
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # 1. Ensure both atoms of the double bond are included
            if begin_idx not in atoms_to_use_set:
                atoms_to_use.append(begin_idx)
                atoms_to_use_set.add(begin_idx)
                symbol_replacements.append(
                    (
                        begin_idx,
                        get_strict_smarts_for_atom(
                            mol.GetAtomWithIdx(begin_idx), use_stereochemistry
                        ),
                    )
                )
            if end_idx not in atoms_to_use_set:
                atoms_to_use.append(end_idx)
                atoms_to_use_set.add(end_idx)
                symbol_replacements.append(
                    (
                        end_idx,
                        get_strict_smarts_for_atom(
                            mol.GetAtomWithIdx(end_idx), use_stereochemistry
                        ),
                    )
                )

            # 2. Ensure each atom of the double bond has a neighbor in the fragment
            for bond_atom_idx in [begin_idx, end_idx]:
                other_bond_atom_idx = (
                    end_idx if bond_atom_idx == begin_idx else begin_idx
                )
                bond_atom = mol.GetAtomWithIdx(bond_atom_idx)

                has_neighbor_in_fragment = False
                for neighbor in bond_atom.GetNeighbors():
                    if (
                        neighbor.GetIdx() != other_bond_atom_idx
                        and neighbor.GetIdx() in atoms_to_use_set
                    ):
                        has_neighbor_in_fragment = True
                        break

                if not has_neighbor_in_fragment:
                    # Add the first available neighbor as a wildcard
                    for neighbor in bond_atom.GetNeighbors():
                        if neighbor.GetIdx() != other_bond_atom_idx:
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx not in atoms_to_use_set:
                                atoms_to_use.append(neighbor_idx)
                                atoms_to_use_set.add(neighbor_idx)
                                symbol_replacements.append(
                                    (neighbor_idx, convert_atom_to_wildcard(neighbor))
                                )
                            break  # only need to add one

    return atoms_to_use, symbol_replacements


def get_frag_around_tetrahedral_center(mol: Chem.Mol, idx: int) -> str:
    """
    Build a molecular fragment around a tetrahedral center, preserving chirality.

    Creates a fragment containing the specified tetrahedral center atom and all its
    immediate neighbors. Uses isotope labels to track mapped atoms and preserve
    their stereochemical configuration in the resulting SMILES string.

    Args:
        mol (Chem.Mol): The molecule containing the tetrahedral center. Must have
            isotope labels already assigned to mapped atoms.
        idx (int): Atom index of the tetrahedral center.

    Returns:
        str: SMILES string of the fragment with explicit chirality, bonds, and hydrogens.

    Note:
        This function uses a two-pass approach to preserve chirality: first extracting
        an initial fragment to capture chiral tags from RDKit's SMILES output, then
        reconstructing atom symbols with isotope and chirality information. This is
        necessary because RDKit cleans chiral tags when obtained from the atom object
        directly versus from the SMILES representation.
    """
    ids_to_include: List[int] = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())

    # Get an initial smiles string for the fragment of the molecule --
    # This string is used to get the correct chirality for the final, more general
    # returned fragment.
    # Likely a better way to do this, but chiral tags seem to be cleaned
    # by rdkit when obtained from a mol vs. from each atom individually
    init_frag = Chem.MolFragmentToSmiles(
        mol,
        ids_to_include,
        isomericSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )
    # assuming that for this task, only the chirality of mapped atoms in the fragment
    # matters -- need to double check this
    # Get a list of each mapped atom token in the fragment
    mapped_atom_tokens = re.findall(r"\[[0-9]+.*?:[0-9]+\]", init_frag)
    map_num_to_chi = {
        int(re.findall(r"(?<=\[)[0-9]+", token)[0]): ("").join(
            re.findall("@+H?", token)
        )
        for token in mapped_atom_tokens
    }
    symbols = []
    for a in mol.GetAtoms():
        at_tag = ""
        chi_tag = ""
        iso_tag = None
        if a.GetIsotope() != 0:
            at_tag = a.GetSymbol()
            iso_tag = a.GetIsotope()

            if iso_tag in map_num_to_chi:
                chi_tag = map_num_to_chi[iso_tag]
        else:
            at_tag = "#{}".format(a.GetAtomicNum())

        if iso_tag is not None:
            symbol = "[{}{}{}]".format(iso_tag, at_tag, chi_tag)
        else:
            symbol = "[{}{}]".format(at_tag, chi_tag)
        symbols.append(symbol)

    return rdmolfiles.MolFragmentToSmiles(
        mol,
        ids_to_include,
        isomericSmiles=True,
        atomSymbols=symbols,
        allBondsExplicit=True,
        allHsExplicit=True,
    )


def check_tetrahedral_centers_equivalent(atom1: Chem.Atom, atom2: Chem.Atom) -> bool:
    """
    Check if two tetrahedral centers have equivalent chirality configurations.

    Determines whether two atoms from potentially different molecules represent
    the same tetrahedral stereochemical configuration, ignoring explicit ChiralTag
    values. This is done by extracting a fragment around atom1 and checking if
    atom2 matches within that fragment using substructure matching with chirality.

    Args:
        atom1 (Chem.Atom): The reference tetrahedral center atom. Its owning molecule
            must have isotope labels assigned to mapped atoms.
        atom2 (Chem.Atom): The atom to compare against atom1. Its owning molecule
            must also have isotope labels assigned to mapped atoms.

    Returns:
        bool: True if atom2 is part of a substructure match indicating equivalent
            chirality to atom1; False otherwise or if fragment parsing fails.
    """
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


def get_changed_atoms(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> Tuple[List[Chem.Atom], List[int], int]:
    """
    Identify atoms that change during a reaction by comparing reactants and products.

    Examines atom-mapped atoms across reactants and products to determine which atoms
    undergo chemical or stereochemical changes. This includes atoms that are leaving
    groups (appear only in reactants), atoms whose properties differ between reactants
    and products, and atoms involved in stereochemical changes (tetrahedral chirality
    or double bond stereochemistry) that are adjacent to the reaction center.

    Args:
        reactants (List[Chem.Mol]): List of reactant molecules with atom map numbers.
        products (List[Chem.Mol]): List of product molecules with atom map numbers.

    Returns:
        Tuple[List[Chem.Atom], List[int], int]:
            - changed_atoms: List of reactant atoms that undergo changes.
            - changed_atom_tags: List of atom map numbers corresponding to changed atoms.
            - err: Error code (0 if successful, non-zero if an error occurred).
    """

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)

    # Find differences
    changed_atoms: List[Chem.Atom] = []  # actual reactant atom species
    changed_atom_tags: List[int] = []  # atom map numbers of those atoms

    prod_tag_counts = Counter(prod_atom_tags)
    prod_atom_by_tag: Dict[int, Chem.Atom] = {}
    for atom, tag in zip(prod_atoms, prod_atom_tags):
        if tag not in prod_atom_by_tag:
            prod_atom_by_tag[tag] = atom

    reac_atom_by_tag: Dict[int, Chem.Atom] = {}
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
        if prod_tag_counts[tag] > 1 or atoms_are_different(
            prod_atom, reac_atom, skip_smarts_check=True, check_local_stereo=True
        ):
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
                # a random specification (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    neighbor_map_num = neighbor.GetAtomMapNum()
                    if neighbor_map_num in changed_atom_tags:
                        tetra_adj_to_rxn = True
                        break
                if tetra_adj_to_rxn:
                    changed_atom_tags.append(atom_tag)
                    changed_atoms.append(ar)

    # Atoms involved in double bonds where stereochemistry changes
    stereo_double_bonds = get_stereogenic_double_bonds(reactants, products)
    for bond_map_nums in stereo_double_bonds:
        for map_num in bond_map_nums:
            if map_num not in changed_tag_set:
                # Find the atom with this map number in the reactants
                atom_to_add = reac_atom_by_tag.get(map_num)
                if atom_to_add:
                    changed_atoms.append(atom_to_add)
                    changed_atom_tags.append(map_num)
                    changed_tag_set.add(map_num)

    return changed_atoms, changed_atom_tags, err


def get_special_groups(mol: Chem.Mol) -> List[Tuple[List[int], List[int]]]:
    """
    Identify special functional groups in a molecule that should be included
    together as a unit during template extraction.

    Searches the molecule for predefined special group patterns (e.g.,
    nitro groups, conjugated systems) and returns the atom indices for each
    matched group. This is used during reactant template extraction to ensure
    chemically meaningful fragments are kept together.

    Args:
        mol (Chem.Mol): The RDKit molecule to search for special groups.

    Returns:
        List[Tuple[List[int], List[int]]]: A list of tuples, where each tuple
            contains:
            - First element (List[int]): Atom indices that trigger inclusion of
              the entire group when matched ("important" atoms).
            - Second element (List[int]): All atom indices in the matched group.

    Note:
        This should only be applied to reactants, not products. Applying to
        products can cause mapping mismatches. The function distinguishes between
        "important" atoms (which trigger group inclusion) and "unimportant"
        atoms (which are only included if an important atom in the same group
        is matched).
    """

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
    """
    Expand a list of atom indices to include nearest neighbors and functional groups.

    Iterates over atoms already marked for inclusion and expands the selection by:
    (1) adding all atoms from any functional group containing a marked atom, and
    (2) evaluating each nearest neighbor for inclusion via expand_atoms_to_use_atom.
    Neighbors that are added are converted to wildcard SMARTS symbols for generalizability.

    Args:
        mol (Chem.Mol): The RDKit molecule containing the atoms.
        atoms_to_use (List[int]): Atom indices already selected for inclusion.
        groups (Optional[List[Any]]): Functional group definitions where each group
            is typically a tuple of (match_atoms, include_atoms). Atoms in match_atoms
            trigger inclusion of all atoms in include_atoms.
        symbol_replacements (Optional[List[Tuple[int, str]]]): Existing list of
            (atom_idx, smarts_symbol) pairs for wildcard replacements. Modified in-place.

    Returns:
        Tuple[List[int], List[Tuple[int, str]]]:
            - Expanded list of atom indices to include.
            - Updated list of symbol replacement tuples for wildcard atoms.
    """

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
                    if idx not in new_atoms_to_use:
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
    """
    Extend the list of atoms to use by considering a candidate atom extension.

    Given an RDKit molecule and a list of atom indices that should be included
    in the reaction, this function extends the list by processing a candidate
    atom. If the atom belongs to a special functional group, all atoms in that
    group are added. Otherwise, the single atom is added with a SMARTS wildcard
    replacement.

    Args:
        mol (Chem.Mol): The RDKit molecule containing the atoms.
        atoms_to_use (List[int]): List of atom indices already included in the reaction.
        atom_idx (int): Index of the candidate atom to consider for extension.
        groups (Optional[List[Tuple[List[int], List[int]]]]): List of special functional
            groups where each group is a tuple of (match_atoms, include_atoms). If the
            candidate atom is in match_atoms, all include_atoms are added.
        symbol_replacements (Optional[List[Tuple[int, str]]]): List of (atom_idx, smarts)
            tuples for SMARTS wildcard replacements. Modified in-place when atoms are added.

    Returns:
        Tuple[List[int], List[Tuple[int, str]]]:
            - Updated list of atom indices to use (may contain new atoms).
            - Updated list of symbol replacements for SMARTS generation.

    Note:
        Groups have highest priority - if an atom matches any group, all atoms in that
        group's include list are added. If the atom is already in atoms_to_use, it is
        skipped unless it triggers a group expansion.
    """
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
    """
    Convert an RDKit atom to a generalized wildcard SMARTS string.

    Uses heuristic generalization rules to create a SMARTS pattern that matches
    chemically equivalent atoms. Terminal atoms (degree == 1) are handled
    differently from non-terminal atoms to preserve hydrogen count information.
    Atom mapping labels from the original SMARTS are preserved in the output.

    Args:
        atom (Chem.Atom): The RDKit atom to convert to a wildcard pattern.

    Returns:
        str: A SMARTS string representing the generalized atom wildcard.
            For terminal atoms: includes symbol, degree (D1), H-count, and charge.
            For non-terminal atoms: includes atomic number (or C/c for carbon),
            aromaticity flag, and charge. Atom mapping labels are preserved.

    Note:
        Carbon atoms receive special handling: aliphatic carbons use 'C',
        aromatic carbons use 'c', while non-carbon atoms use '#{atomic_num}'
        notation with an optional 'a' flag for aromaticity.
    """

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
    label = re.search(r"\:[0-9]+\]", atom.GetSmarts())
    if label:
        symbol += label.group()
    else:
        symbol += "]"

    return symbol


def reassign_atom_mapping(transform: str) -> str:
    """
    Reassign atom-mapping labels in a reaction SMILES from left to right.

    Takes an atom-mapped reaction SMILES and reassigns the atom-mapping labels
    (numbers) sequentially from left to right, starting from 1. Labels are
    reassigned based on their first occurrence in the string, maintaining the
    mapping consistency between reactants and products.

    Args:
        transform (str): An atom-mapped reaction SMILES string with labels
            in the format ":N]" where N is the atom-mapping number.

    Returns:
        str: The reaction SMILES with reassigned atom-mapping labels.

    Example:
        >>> reassign_atom_mapping("[C:2][O:1]>>[C:1][O:2]")
        '[C:1][O:2]>>[C:1][O:2]'
    """

    all_labels: List[str] = re.findall(r"\:([0-9]+)\]", transform)

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
        r"\:[0-9]+\]", lambda match: ":" + replacements.pop(0) + "]", transform
    )

    return transform_newmaps


def get_strict_smarts_for_atom(
    atom: Chem.Atom, use_stereochemistry: bool = True
) -> str:
    """
    Generate a SMARTS pattern that matches an RDKit atom as strictly as possible.

    The pattern includes explicit atom properties: atomic symbol, stereochemistry,
    total hydrogen count, degree (number of explicit connections), and formal charge.

    Args:
        atom (Chem.Atom): The RDKit atom object to generate a SMARTS for.
        use_stereochemistry (bool): If True, include the atom's chiral tag (@ or @@)
            in the SMARTS pattern. Defaults to True.

    Returns:
        str: A SMARTS pattern string that strictly matches atoms with the same
            properties as the input atom.
    """

    symbol = atom.GetSmarts()
    if atom.GetSymbol() == "H":
        symbol = "[#1]"

    if "[" not in symbol:
        symbol = "[" + symbol + "]"

    # Explicit stereochemistry - *before* H
    if use_stereochemistry:
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
    else:
        symbol = symbol.replace("@@", "").replace("@", "")

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
    changed_atom_tags: List[int], reactant_fragments: str
) -> List[int]:
    """
    Expand the list of changed atom tags with additional tags found in reactant fragments.

    Given a list of changed atom tags and a reactant fragments string, this function
    identifies atom tags present in the reactant fragments (using the pattern ':<number>]')
    and returns those tags which are not already present in the changed_atom_tags list.
    This ensures that tagged atoms from the reactant side are included in the products.

    Args:
        changed_atom_tags (List[int]): List of atom tag numbers already identified as changed.
        reactant_fragments (str): SMILES-like string containing atom tags (e.g., '[*+:1]') from the reactant side.

    Returns:
        List[int]: List of atom tag numbers found in reactant_fragments that are not already in changed_atom_tags.

    Raises:
        ValueError: If an atom tag extracted from reactant_fragments is not a valid digit string.
    """

    expansion = []
    atom_tags_in_reactant_fragments: List[str] = re.findall(
        r"\:([0-9]+)\]", reactant_fragments
    )
    for atom_tag in atom_tags_in_reactant_fragments:
        if not atom_tag.isdigit():
            raise ValueError(f"Invalid atom tag: {atom_tag}")
        map_num = int(atom_tag)
        if map_num not in changed_atom_tags:
            expansion.append(map_num)
    return expansion


def get_fragments_for_changed_atoms(
    mols: List[Chem.Mol],
    changed_atom_tags: List[int],
    radius: int = 0,
    category: str = "reactants",
    expansion: Optional[List[int]] = None,
    no_special_groups: bool = False,
    include_all_unmapped_reactant_atoms: bool = True,
    use_stereochemistry: bool = True,
) -> Tuple[str, bool, bool]:
    """
    Extract molecular fragments around changed atoms from a list of RDKit molecules.

    For each molecule, identifies atoms marked by the provided tags, expands the
    selection by the specified radius, and generates SMILES strings using
    MolFragmentToSmiles. Handles stereochemistry preservation, special reactive
    groups, and unmapped atom inclusion based on the category (reactants or products).

    Args:
        mols (List[Chem.Mol]): List of RDKit molecules to extract fragments from.
        changed_atom_tags (List[int]): List of atom map tags identifying changed atoms.
        radius (int): Number of bond hops to expand from changed atoms. Defaults to 0.
        category (str): Molecule category, either "reactants" or "products". Defaults to "reactants".
        expansion (Optional[List[int]]): Additional atom tags to include in product fragments
            that should be generalized as wildcards. Defaults to None.
        no_special_groups (bool): If True, skip detection of special reactive groups
            (reactants only). Defaults to False.
        include_all_unmapped_reactant_atoms (bool): If True, include all unmapped atoms
            in reactant fragments. Defaults to True.
        use_stereochemistry (bool): If True, preserve stereochemistry in fragment SMILES
            and perform chirality mismatch correction. Defaults to True.

    Returns:
        Tuple[str, bool, bool]:
            - SMILES string of extracted fragments, concatenated with dots (periods removed
              from individual fragments).
            - Boolean indicating if this is an intramolecular reaction (single changed molecule).
            - Boolean indicating if this is a dimerization (two identical changed molecules).

    Raises:
        ValueError: If an atom tag contains non-digit characters.
        RuntimeError: If MolFragmentToSmiles fails even with isomericSmiles=False.
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
            atom_smarts = atom.GetSmarts()
            # Check self (only tagged atoms)
            if ":" in atom_smarts:
                atom_tag = atom_smarts.split(":")[1][:-1]
                if not atom_tag.isdigit():
                    raise ValueError(f"Invalid atom tag: {atom_tag}")
                atom_tag_int = int(atom_tag)
                if atom_tag_int in changed_atom_tags:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom, use_stereochemistry)
                    if symbol != atom_smarts:
                        symbol_replacements.append((atom.GetIdx(), symbol))
                    continue

        # Fully define leaving groups and this molecule participates?
        if include_all_unmapped_reactant_atoms and len(atoms_to_use) > 0:
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

        # After initial expansion, ensure stereo double bonds are complete
        atoms_to_use, symbol_replacements = ensure_complete_stereo_double_bonds(
            mol,
            atoms_to_use,
            symbol_replacements,
            use_stereochemistry=use_stereochemistry,
        )

        if category == "products":
            # Add extra labels to include (for products only)
            if expansion:
                for atom in mol.GetAtoms():
                    if ":" not in atom.GetSmarts():
                        continue
                    label = atom.GetSmarts().split(":")[1][:-1]
                    if not label.isdigit():
                        raise ValueError(f"Invalid atom tag: {label}")
                    label_int = int(label)
                    if label_int in expansion and label_int not in changed_atom_tags:
                        atoms_to_use.append(atom.GetIdx())
                        # Make the expansion a wildcard
                        symbol_replacements.append(
                            (atom.GetIdx(), convert_atom_to_wildcard(atom))
                        )

            # Make sure unmapped atoms are included (from products)
            for atom in mol.GetAtoms():
                if not atom.GetAtomMapNum():
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom, use_stereochemistry)
                    symbol_replacements.append((atom.GetIdx(), symbol))

        if not atoms_to_use:
            continue

        atoms_to_use = list(dict.fromkeys(atoms_to_use))

        # Define new symbols based on symbol_replacements
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for i, symbol in symbol_replacements:
            symbols[i] = symbol

        mol_without_map_nums = Chem.Mol(mol)
        for a in mol_without_map_nums.GetAtoms():
            a.SetAtomMapNum(0)

        try:
            this_fragment = rdmolfiles.MolFragmentToSmiles(
                mol_without_map_nums,
                atoms_to_use,
                atomSymbols=symbols,
                allHsExplicit=True,
                isomericSmiles=use_stereochemistry,
                allBondsExplicit=True,
            )
        except RuntimeError:
            try:
                this_fragment = rdmolfiles.MolFragmentToSmiles(
                    mol_without_map_nums,
                    atoms_to_use,
                    atomSymbols=symbols,
                    allHsExplicit=True,
                    isomericSmiles=False,
                    allBondsExplicit=True,
                )
            except RuntimeError as e2:
                raise e2

        if use_stereochemistry:
            this_fragment_mol = rdmolfiles.MolFromSmarts(this_fragment)
            if this_fragment_mol is not None:
                src_map = {
                    a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomMapNum()
                }
                needs_correction = False
                for frag_atom in this_fragment_mol.GetAtoms():
                    mapnum = frag_atom.GetAtomMapNum()
                    if not mapnum:
                        continue
                    if (
                        frag_atom.GetChiralTag()
                        == Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                    ):
                        continue
                    src_atom = src_map.get(mapnum)
                    if (
                        src_atom is None
                        or src_atom.GetChiralTag()
                        == Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                    ):
                        continue
                    if atom_chirality_matches(frag_atom, src_atom) == -1:
                        src_idx = src_atom.GetIdx()
                        prev = symbols[src_idx]
                        if "@@" in prev:
                            symbols[src_idx] = prev.replace("@@", "@")
                        elif "@" in prev:
                            symbols[src_idx] = prev.replace("@", "@@")
                        needs_correction = True

                if needs_correction:
                    try:
                        this_fragment = rdmolfiles.MolFragmentToSmiles(
                            mol_without_map_nums,
                            atoms_to_use,
                            atomSymbols=symbols,
                            allHsExplicit=True,
                            isomericSmiles=use_stereochemistry,
                            allBondsExplicit=True,
                        )
                    except RuntimeError:
                        pass

        fragments += "(" + this_fragment + ")."

    # auxiliary template information: is this an intramolecular reaction or dimerization?
    mol_copy = Chem.Mol(mol)
    mols_changed.append(Chem.MolToSmiles(clear_mapnum(mol_copy), True))

    intra_only = 1 == len(mols_changed)
    dimer_only = (1 == len(set(mols_changed))) and (len(mols_changed) == 2)

    return fragments[:-1], intra_only, dimer_only


def split_reaction_smarts(reaction_smarts: str) -> List[str]:
    """
    Split a multi-component reaction SMARTS into individual component reactions.

    Parses a reaction SMARTS containing multiple reactants and/or products,
    then splits it into separate reaction SMARTS strings where each reactant
    is paired only with the product(s) that share atom map numbers with it.

    Args:
        reaction_smarts (str): An atom-mapped reaction SMARTS string in the
            format "reactant1.react2>>product1.prod2".

    Returns:
        List[str]: A list of split reaction SMARTS strings, where each string
            contains one reactant and its associated product(s) based on
            shared atom map numbers.
    """
    reactants = reaction_smarts.split(">>")[0]
    products = reaction_smarts.split(">>")[1]

    reactant_str_mols_mapping = {
        reactant: Chem.MolFromSmarts(reactant) for reactant in reactants.split(".")
    }
    product_str_mols_mapping = {
        product: Chem.MolFromSmarts(product) for product in products.split(".")
    }

    reactant_str_atom_map_nums = {}
    for reactant_str, reactant_mol in reactant_str_mols_mapping.items():
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum():
                if reactant_str not in reactant_str_atom_map_nums:
                    reactant_str_atom_map_nums[reactant_str] = {atom.GetAtomMapNum()}
                else:
                    reactant_str_atom_map_nums[reactant_str].add(atom.GetAtomMapNum())

    product_str_atom_map_nums = {}
    for product_str, product_mol in product_str_mols_mapping.items():
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum():
                if product_str not in product_str_atom_map_nums:
                    product_str_atom_map_nums[product_str] = {atom.GetAtomMapNum()}
                else:
                    product_str_atom_map_nums[product_str].add(atom.GetAtomMapNum())

    split_smarts = []
    for reactant_str, reactant_map_nums in reactant_str_atom_map_nums.items():
        product_smarts = []
        for product_str, product_map_nums in product_str_atom_map_nums.items():
            if not product_map_nums.isdisjoint(reactant_map_nums):
                product_smarts.append(product_str)

        if product_smarts:
            split_smarts.append(f"{reactant_str}>>{'.'.join(product_smarts)}")

    return split_smarts


def canonicalize_transform(transform: str, fix_cycle_chirality: bool = True) -> str:
    """
    Convert an atom-mapped SMARTS transform to a canonical form.

    The transform is canonicalized by reordering reactant and product templates
    based on their unmapped representations and reassigning atom-mapping labels
    sequentially. Optionally, chirality around unmapped ring closures can be
    inverted to handle stereochemical invariants.

    Args:
        transform (str): An atom-mapped reaction SMARTS string in the form
            "reactants>>products".
        fix_cycle_chirality (bool): If True, invert tetrahedral chirality for
            unmapped atoms preceding ring closure tokens to handle stereochemical
            invariants in cyclic templates.

    Returns:
        str: The canonicalized reaction SMARTS with reordered templates and
            sequentially reassigned atom-mapping labels.
    """

    transform_reordered = ">>".join(
        [_canonicalize_template(x) for x in transform.split(">>")]
    )

    if fix_cycle_chirality:
        transform_reordered = invert_chirality_around_unmapped_ring_closure(
            transform_reordered
        )

    return reassign_atom_mapping(transform_reordered)


def _canonicalize_template(template: str) -> str:
    """
    Canonicalize a template SMARTS string by sorting fragments and molecules.

    This function takes one-half of a template SMARTS string (i.e., reactants or
    products) and re-orders fragments within molecules and molecules themselves
    based on lexicographic ordering of an equivalent string without atom mapping.
    This ensures consistent template representation regardless of input order.

    Args:
        template (str): A template SMARTS string representing one side of a
            reaction (reactants or products). Expected format wraps molecules
            in parentheses and separates them with ').(', with fragments within
            molecules separated by '.' and atom mapping labels like ':N]'.

    Returns:
        str: The canonicalized template with fragments sorted within each molecule
            and molecules sorted overall, maintaining original atom mapping labels.
    """

    # Strip labels to get sort orders
    template_nolabels = re.sub(r"\:[0-9]+\]", "]", template)

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
    reaction: rdChiralTemplateExtractInput,
    no_special_groups: bool = False,
    radius: int = 1,
    use_stereochemistry: bool = True,
    canonicalize_template: bool = True,
    maximum_number_unmapped_product_atoms: int = 5,
    include_all_unmapped_reactant_atoms: bool = True,
) -> ExtractedTemplate:
    """
    Extract a retrosynthetic reaction template from a mapped chemical reaction.

    This function analyzes a chemical reaction with atom-mapped reactants and products
    to identify the reaction center and extract a template representing the
    transformation. The template captures the local environment around atoms that
    change during the reaction, including bonds formed, broken, or modified.

    Args:
        reaction (rdChiralTemplateExtractInput): Dictionary containing the reaction
            data with keys 'reactants' (SMILES string), 'products' (SMILES string),
            and '_id' (optional reaction identifier).
        no_special_groups (bool): If True, disable special functional group handling
            during fragment extraction. Defaults to False.
        radius (int): Number of bonds to expand around changed atoms when extracting
            fragments. A larger radius captures more context. Defaults to 1.
        use_stereochemistry (bool): If True, include stereochemical information in
            the extracted template. Defaults to True.
        canonicalize_template (bool): If True, canonicalize the reaction SMARTS
            string for consistent representation. Defaults to True.
        maximum_number_unmapped_product_atoms (int): Maximum allowed number of
            unmapped atoms in the products. Reactions exceeding this limit are
            skipped. Defaults to 5.
        include_all_unmapped_reactant_atoms (bool): If True, include all unmapped
            reactant atoms in the template, not just those near the reaction center.
            Defaults to True.

    Returns:
        ExtractedTemplate: A dictionary containing the extracted template with the
            following keys:
            - 'products': Product fragment SMARTS string
            - 'reactants': Reactant fragment SMARTS string
            - 'spectators': SMILES string of spectator molecules (reactants not
              participating in the reaction)
            - 'reaction_smarts': Complete retrosynthetic reaction SMARTS
              (products>>reactants)
            - 'separated_reaction_smarts': List of individual reaction SMARTS
              components
            - 'intra_only': Boolean indicating if the reaction is intramolecular
            - 'dimer_only': Boolean indicating if the reaction involves dimerization
            - 'reaction_id': The reaction identifier from the input
            - 'necessary_reagent': SMILES fragment for unmapped product atoms that
              must be supplied as reagents

    Raises:
        ValueError: Propagated from get_fragments_for_changed_atoms if fragment
            extraction fails (caught internally and returns default template).

    Note:
        This function returns a default empty template (with empty strings and
        False flags) if any of the following conditions occur:
        - RDKit fails to parse reactants or products
        - The number of unmapped product atoms exceeds the maximum threshold
        - No reactant atoms are mapped to product atoms
        - Molecule sanitization fails
        - No atoms change between reactants and products
        - The extracted reaction SMARTS fails RDKit validation
        The returned template uses retro-synthetic direction (products>>reactants).
    """
    # Build a default template that preserves the reaction_id for early returns
    default_template: ExtractedTemplate = {
        **DEFAULT_EXTRACTED_TEMPLATE,
        "reaction_id": reaction["_id"],
    }

    reactants = mols_from_smiles_list(
        replace_deuterated(reaction["reactants"]).split(".")
    )
    products = mols_from_smiles_list(
        replace_deuterated(reaction["products"]).split(".")
    )

    # if rdkit cant understand molecule, return
    if None in reactants:
        return default_template
    if None in products:
        return default_template

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
                if num_unmapped_product_atoms > maximum_number_unmapped_product_atoms:
                    # Skip this example - too many unmapped product atoms!
                    return default_template
        if num_mapped_atoms < len(prod_atoms):
            are_unmapped_product_atoms = True

    reactants_in_reaction = []
    for reactant in reactants:
        for atom in reactant.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num and map_num in seen_atom_map_nums:
                reactants_in_reaction.append(reactant)
                break
    spectators_string = ".".join(
        [
            Chem.MolToSmiles(reactant)
            for reactant in reactants
            if reactant not in reactants_in_reaction
        ]
    )
    reactants = reactants_in_reaction
    if not reactants:
        return default_template

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
        return default_template

    if None in reactants + products:
        return default_template

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
                        isomericSmiles=use_stereochemistry,
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
        return default_template
    if not changed_atom_tags:
        return default_template

    try:
        # Get fragments for reactants
        reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(
            reactants,
            changed_atom_tags,
            radius=radius,
            expansion=[],
            category="reactants",
            no_special_groups=no_special_groups,
            include_all_unmapped_reactant_atoms=include_all_unmapped_reactant_atoms,
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
        return default_template

    # Put together and canonicalize (as best as possible)
    rxn_string = "{}>>{}".format(reactant_fragments, product_fragments)
    if canonicalize_template:
        rxn_canonical = canonicalize_transform(rxn_string)
    else:
        rxn_canonical = rxn_string
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
        return default_template
    if rxn.Validate()[1] != 0:
        return default_template

    template: ExtractedTemplate = {
        "products": products_string,
        "reactants": reactants_string,
        "spectators": spectators_string,
        "reaction_smarts": retro_canonical,
        "separated_reaction_smarts": split_reaction_smarts(retro_canonical),
        "intra_only": intra_only,
        "dimer_only": dimer_only,
        "reaction_id": reaction["_id"],
        "necessary_reagent": extra_reactant_fragment,
    }

    return template


def extract_from_reaction_smiles(
    rxn_smiles: str,
    no_special_groups: bool = False,
    radius: int = 1,
    use_stereochemistry: bool = True,
    canonicalize_template: bool = True,
    maximum_number_unmapped_product_atoms: int = 5,
    include_all_unmapped_reactant_atoms: bool = True,
    reaction_id: Optional[str | int] = None,
) -> ExtractedTemplate:
    """
    Extract a retrosynthetic reaction template from a reaction SMILES string.

    This function parses a reaction SMILES string and extracts a template representing
    the chemical transformation. It is a convenience wrapper around extract_from_reaction
    that handles the SMILES parsing and delegates to the core extraction logic.

    Args:
        rxn_smiles (str): Reaction SMILES string in the format "reactants>>products".
            The reactants and products should have atom mapping numbers for template
            extraction to work correctly.
        no_special_groups (bool): If True, disable special functional group handling
            during fragment extraction. Defaults to False.
        radius (int): Number of bonds to expand around changed atoms when extracting
            fragments. A larger radius captures more context. Defaults to 1.
        use_stereochemistry (bool): If True, include stereochemical information in
            the extracted template. Defaults to True.
        canonicalize_template (bool): If True, canonicalize the reaction SMARTS
            string for consistent representation. Defaults to True.
        maximum_number_unmapped_product_atoms (int): Maximum allowed number of
            unmapped atoms in the products. Reactions exceeding this limit are
            skipped. Defaults to 5.
        include_all_unmapped_reactant_atoms (bool): If True, include all unmapped
            reactant atoms in the template, not just those near the reaction center.
            Defaults to True.
        reaction_id (Optional[str | int]): Optional identifier for the reaction.
            This will be included in the returned template.

    Returns:
        ExtractedTemplate: A dictionary containing the extracted template with the
            following keys:
            - 'products': Product fragment SMARTS string
            - 'reactants': Reactant fragment SMARTS string
            - 'spectators': SMILES string of spectator molecules (reactants not
              participating in the reaction)
            - 'reaction_smarts': Complete retrosynthetic reaction SMARTS
              (products>>reactants)
            - 'separated_reaction_smarts': List of individual reaction SMARTS
              components
            - 'intra_only': Boolean indicating if the reaction is intramolecular
            - 'dimer_only': Boolean indicating if the reaction involves dimerization
            - 'reaction_id': The reaction identifier from the input
            - 'necessary_reagent': SMILES fragment for unmapped product atoms that
              must be supplied as reagents

    Raises:
        ValueError: If the reaction SMILES does not contain exactly one '>>' separator.
    """
    if len(rxn_smiles.split(">>")) != 2:
        raise ValueError("Reaction SMILES must contain exactly one >>")

    reactants = rxn_smiles.split(">>")[0]
    products = rxn_smiles.split(">>")[1]
    reaction_dict: rdChiralTemplateExtractInput = {
        "reactants": reactants,
        "products": products,
        "_id": reaction_id,
    }

    return extract_from_reaction(
        reaction=reaction_dict,
        no_special_groups=no_special_groups,
        radius=radius,
        use_stereochemistry=use_stereochemistry,
        canonicalize_template=canonicalize_template,
        maximum_number_unmapped_product_atoms=maximum_number_unmapped_product_atoms,
        include_all_unmapped_reactant_atoms=include_all_unmapped_reactant_atoms,
    )
