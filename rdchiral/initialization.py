from typing import Dict, List, Tuple

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import rdChemReactions, rdmolops
from rdkit.Chem.rdchem import BondDir, BondStereo, ChiralType

from rdchiral.bonds import (
    bond_dirs_by_mapnum,
    enumerate_possible_cistrans_defs,
    get_atoms_across_double_bonds,
)
from rdchiral.chiral import template_atom_could_have_been_tetra

BondDirOpposite = {
    AllChem.BondDir.ENDUPRIGHT: AllChem.BondDir.ENDDOWNRIGHT,
    AllChem.BondDir.ENDDOWNRIGHT: AllChem.BondDir.ENDUPRIGHT,
}


class rdchiralReaction(object):
    """
    Class to store everything that should be pre-computed for a reaction. This
    makes library application much faster, since we can pre-do a lot of work
    instead of doing it for every mol-template pair

    Attributes:
        reaction_smarts (str): Reaction SMARTS string
        rxn (rdkit.Chem.rdChemReactions.ChemicalReaction): RDKit reaction object.
            Generated from `reaction_smarts` using `initialize_rxn_from_smarts`
        template_r: Reaction reactant template fragments
        template_p: Reaction product template fragments
        atoms_rt_map (dict): Dictionary mapping from atom map number to RDKit Atom for reactants
        atoms_pt_map (dict): Dictionary mapping from atom map number to RDKit Atom for products
        atoms_rt_idx_to_map (dict): Dictionary mapping from atom idx to RDKit Atom for reactants
        atoms_pt_idx_to_map (dict): Dictionary mapping from atom idx to RDKit Atom for products

    Args:
        reaction_smarts (str): Reaction SMARTS string
    """

    def __init__(self, reaction_smarts: str):
        # Keep smarts, useful for reporting
        self.reaction_smarts: str = reaction_smarts

        # Initialize - assigns stereochemistry and fills in missing rct map numbers
        self.rxn: rdChemReactions.ChemicalReaction = initialize_rxn_from_smarts(
            reaction_smarts
        )

        # Combine template fragments so we can play around with mapnums
        template_r, template_p = _get_template_frags_from_rxn(self.rxn)
        self.template_r_orig: Chem.Mol = Chem.Mol(template_r)
        self.template_p_orig: Chem.Mol = Chem.Mol(template_p)
        self.template_r: Chem.Mol = Chem.Mol(self.template_r_orig)
        self.template_p: Chem.Mol = Chem.Mol(self.template_p_orig)

        # Define molAtomMapNumber->atom dictionary for template rct and prd
        self.atoms_rt_map: Dict[int, Chem.Atom] = {
            a.GetAtomMapNum(): a
            for a in self.template_r.GetAtoms()
            if a.GetAtomMapNum()
        }
        self.atoms_pt_map: Dict[int, Chem.Atom] = {
            a.GetAtomMapNum(): a
            for a in self.template_p.GetAtoms()
            if a.GetAtomMapNum()
        }

        # Back-up the mapping for the reaction
        self.atoms_rt_idx_to_map: Dict[int, int] = {
            a.GetIdx(): a.GetAtomMapNum() for a in self.template_r_orig.GetAtoms()
        }
        self.atoms_pt_idx_to_map: Dict[int, int] = {
            a.GetIdx(): a.GetAtomMapNum() for a in self.template_p_orig.GetAtoms()
        }

        # Check consistency (this should not be necessary...)
        if any(
            self.atoms_rt_map[i].GetAtomicNum() != self.atoms_pt_map[i].GetAtomicNum()
            for i in self.atoms_rt_map
            if i in self.atoms_pt_map
        ):
            raise ValueError("Atomic identity should not change in a reaction!")

        # Call template_atom_could_have_been_tetra to pre-assign value to atom
        for a in self.template_r.GetAtoms():
            template_atom_could_have_been_tetra(a)
        for a in self.template_p.GetAtoms():
            template_atom_could_have_been_tetra(a)

        # Pre-list chiral double bonds (for copying back into outcomes/matching)
        self.rt_bond_dirs_by_mapnum: Dict[Tuple[int, int], BondDir] = (
            bond_dirs_by_mapnum(self.template_r)
        )
        self.pt_bond_dirs_by_mapnum: Dict[Tuple[int, int], BondDir] = (
            bond_dirs_by_mapnum(self.template_p)
        )

        # Enumerate possible cis/trans...
        self.required_rt_bond_defs, self.required_bond_defs_coreatoms = (
            enumerate_possible_cistrans_defs(self.template_r)
        )

        self.template_has_tetra_stereo: bool = _has_tetra_stereo(
            self.template_r
        ) or _has_tetra_stereo(self.template_p)
        self.template_has_doublebond_stereo: bool = _has_doublebond_stereo(
            self.template_r
        ) or _has_doublebond_stereo(self.template_p)
        self.template_is_chiral: bool = (
            self.template_has_tetra_stereo or self.template_has_doublebond_stereo
        )

    def reset(self) -> None:
        """Reset atom map numbers for template fragment atoms"""
        self.template_r = self.template_r_orig
        self.template_p = self.template_p_orig


class rdchiralReactants(object):
    """
    Class to store everything that should be pre-computed for a reactant mol
    so that library application is faster

    Attributes:
        reactant_smiles (str): Reactant SMILES string
        reactants (rdkit.Chem.rdchem.Mol): RDKit Molecule create from `_initialize_reactants_from_smiles`
        atoms_r (dict): Dictionary mapping from atom map number to atom in `reactants` Molecule
        reactants_achiral (rdkit.Chem.rdchem.Mol): achiral version of `reactants`
        bonds_by_mapnum (list): List of reactant bonds
            (int, int, rdkit.Chem.rdchem.Bond)
        bond_dirs_by_mapnum (dict): Dictionary mapping from atom map number tuples to BondDir
        atoms_across_double_bonds (list): List of cis/trans specifications from `get_atoms_across_double_bonds`

    Methods:
        idx_to_mapnum (int): Returns atom map number for given atom idx

    Args:
        reactant_smiles (str): Reactant SMILES string
        custom_reactant_mapping (bool): Whether to use custom reactant mapping
    """

    def __init__(self, reactant_smiles: str, custom_reactant_mapping: bool = False):
        # Keep original smiles, useful for reporting
        self.reactant_smiles: str = reactant_smiles
        self.custom_mapping: bool = custom_reactant_mapping

        # Initialize into RDKit mol
        self.reactants: Chem.Mol = _initialize_reactants_from_smiles(reactant_smiles)

        # Set mapnum->atom dictionary
        # all reactant atoms must be mapped after initialization, so this is safe

        self.atoms_r: Dict[int, Chem.Atom] = {}
        self._idx_to_map_num: Dict[int, int] = {}
        has_tetra_stereo = False
        n_atoms = self.reactants.GetNumAtoms()
        n_bonds = self.reactants.GetNumBonds()
        for idx in range(n_atoms):
            atom = self.reactants.GetAtomWithIdx(idx)
            if not self.custom_mapping:
                atom.SetAtomMapNum(idx + 1)
                map_num = idx + 1
            else:
                map_num = atom.GetAtomMapNum()

            self.atoms_r[map_num] = atom
            self._idx_to_map_num[idx] = map_num
            if (
                not has_tetra_stereo
                and atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
            ):
                has_tetra_stereo = True

        # Create copy of molecule without chiral information, used with
        # RDKit's naive runReactants
        self.reactants_achiral = Chem.Mol(self.reactants)
        for idx in range(n_atoms):
            self.reactants_achiral.GetAtomWithIdx(idx).SetChiralTag(
                ChiralType.CHI_UNSPECIFIED
            )
        for idx in range(n_bonds):
            b = self.reactants_achiral.GetBondWithIdx(idx)
            b.SetStereo(BondStereo.STEREONONE)
            b.SetBondDir(BondDir.NONE)

        # Pre-list reactant bonds (for stitching broken products)
        self.bonds_by_mapnum: List[Tuple[int, int, Chem.Bond]] = []
        self.bond_dirs_by_mapnum: Dict[Tuple[int, int], BondDir] = {}
        has_doublebond_stereo = False
        for bond_idx in range(n_bonds):
            b = self.reactants.GetBondWithIdx(bond_idx)
            i = b.GetBeginAtom().GetAtomMapNum()
            j = b.GetEndAtom().GetAtomMapNum()
            self.bonds_by_mapnum.append((i, j, b))

            bond_dir = b.GetBondDir()
            if bond_dir != BondDir.NONE:
                has_doublebond_stereo = True
                self.bond_dirs_by_mapnum[(i, j)] = bond_dir
                self.bond_dirs_by_mapnum[(j, i)] = BondDirOpposite[bond_dir]

        # Get atoms across double bonds defined by mapnum
        self.atoms_across_double_bonds: List[
            Tuple[Tuple[int, int, int, int], Tuple[BondDir, BondDir], bool]
        ] = get_atoms_across_double_bonds(self.reactants)

        self.reactants_has_tetra_stereo = has_tetra_stereo
        self.reactants_has_doublebond_stereo = has_doublebond_stereo
        self.reactants_is_chiral: bool = (
            self.reactants_has_tetra_stereo or self.reactants_has_doublebond_stereo
        )

    def idx_to_mapnum(self, idx: int) -> int:
        return self._idx_to_map_num[idx]


def initialize_rxn_from_smarts(
    reaction_smarts: str,
) -> rdChemReactions.ChemicalReaction:
    """
    Initialize a reaction from a SMARTS string.

    Args:
        reaction_smarts (str): Reaction SMARTS string.

    Returns:
        rdChemReactions.ChemicalReaction: RDKit reaction object with validation applied.
    """
    # Initialize reaction
    rxn: rdChemReactions.ChemicalReaction = rdChemReactions.ReactionFromSmarts(
        reaction_smarts
    )
    rxn.Initialize()
    if rxn.Validate()[1] != 0:
        raise ValueError("validation failed")

    # Figure out if there are unnecessary atom map numbers (that are not balanced)
    # e.g., leaving groups for retrosynthetic templates. This is because additional
    # atom map numbers in the input SMARTS template may conflict with the atom map
    # numbers of the molecules themselves
    products_vec: rdChemReactions.MOL_SPTR_VECT = rxn.GetProducts()
    products: List[Chem.Mol] = [mol for mol in products_vec]
    prd_maps: List[int] = []
    for prd in products:
        for a in prd.GetAtoms():
            if a.GetAtomMapNum():
                prd_maps.append(a.GetAtomMapNum())

    unmapped = 700
    reactants_vec: rdChemReactions.MOL_SPTR_VECT = rxn.GetReactants()
    reactants: List[Chem.Mol] = [mol for mol in reactants_vec]
    for rct in reactants:
        rct.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(rct)
        # Fill in atom map numbers
        for a in rct.GetAtoms():
            if not a.GetAtomMapNum() or a.GetAtomMapNum() not in prd_maps:
                a.SetAtomMapNum(unmapped)
                unmapped += 1
    if unmapped > 800:
        raise ValueError(
            "Why do you have so many unmapped atoms in the template reactants?"
        )

    return rxn


def _initialize_reactants_from_smiles(reactant_smiles: str) -> Chem.Mol:
    """
    Initialize a reactant molecule from a SMILES string.

    Args:
        reactant_smiles (str): Reactant SMILES string.

    Returns:
        Chem.Mol: RDKit molecule with stereochemistry assigned and its property cache updated.
    """
    # Initialize reactants
    reactants: Chem.Mol = Chem.MolFromSmiles(reactant_smiles)
    Chem.AssignStereochemistry(reactants, flagPossibleStereoCenters=True)
    reactants.UpdatePropertyCache(strict=False)
    return reactants


def _get_template_frags_from_rxn(
    rxn: rdChemReactions.ChemicalReaction,
) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Get template fragments from RDKit reaction object

    Args:
        rxn (rdChemReactions.ChemicalReaction): RDKit reaction object

    Returns:
        (Chem., Chem.): tuple of fragment molecules
    """
    # Copy reaction template so we can play around with map numbers
    reactants_vec: rdChemReactions.MOL_SPTR_VECT = rxn.GetReactants()
    reactants: List[Chem.Mol] = [mol for mol in reactants_vec]
    for i, rct in enumerate(reactants):
        if i == 0:
            template_r = rct
        else:
            template_r = rdmolops.CombineMols(template_r, rct)
    products_vec: rdChemReactions.MOL_SPTR_VECT = rxn.GetProducts()
    products: List[Chem.Mol] = [mol for mol in products_vec]
    for i, prd in enumerate(products):
        if i == 0:
            template_p = prd
        else:
            template_p = rdmolops.CombineMols(template_p, prd)
    return template_r, template_p


def _has_tetra_stereo(mol: Chem.Mol) -> bool:
    """
    Check whether a molecule contains any tetrahedral stereochemistry annotations.

    Args:
        mol (Chem.Mol): RDKit molecule to inspect.

    Returns:
        bool: True if any atom has a chiral tag other than `CHI_UNSPECIFIED`, otherwise False.
    """
    return any(a.GetChiralTag() != ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())


def _has_doublebond_stereo(mol: Chem.Mol) -> bool:
    """
    Check whether a molecule contains any directional bond annotations (commonly used for double-bond stereochemistry).

    Args:
        mol (Chem.Mol): RDKit molecule to inspect.

    Returns:
        bool: True if any bond has a non-NONE `BondDir` value, otherwise False.
    """
    return any(b.GetBondDir() != BondDir.NONE for b in mol.GetBonds())
