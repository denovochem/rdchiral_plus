from typing import Dict, List, Optional, Set, Tuple

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

    def __init__(self, reaction_smarts: str, lazy_init: bool = True):
        # Keep smarts, useful for reporting
        self.reaction_smarts: str = reaction_smarts

        # Initialize - assigns stereochemistry and fills in missing rct map numbers
        self.rxn: rdChemReactions.ChemicalReaction = initialize_rxn_from_smarts(
            reaction_smarts
        )

        self._template_r_orig: Optional[Chem.Mol] = None
        self._template_p_orig: Optional[Chem.Mol] = None
        self._template_r: Optional[Chem.Mol] = None
        self._template_p: Optional[Chem.Mol] = None

        # Define molAtomMapNumber->atom dictionary for template rct and prd
        self._atoms_rt_map: Optional[Dict[int, Chem.Atom]] = None
        self._atoms_pt_map: Optional[Dict[int, Chem.Atom]] = None

        # Back-up the mapping for the reaction
        self._atoms_rt_idx_to_map: Optional[Dict[int, int]] = None
        self._atoms_pt_idx_to_map: Optional[Dict[int, int]] = None

        # Pre-list chiral double bonds (for copying back into outcomes/matching)
        self._rt_bond_dirs_by_mapnum: Optional[Dict[Tuple[int, int], BondDir]] = None
        self._pt_bond_dirs_by_mapnum: Optional[Dict[Tuple[int, int], BondDir]] = None

        # Enumerate possible cis/trans...
        self._required_rt_bond_defs: Optional[
            Dict[Tuple[int, int, int, int], Tuple[BondDir, BondDir]]
        ] = None
        self._required_bond_defs_coreatoms: Optional[Set[Tuple[int, int]]] = None

        self._template_has_tetra_stereo: Optional[bool] = None
        self._template_has_doublebond_stereo: Optional[bool] = None
        self._template_is_chiral: Optional[bool] = None

        if not lazy_init:
            self._ensure_templates()
            _ = self.template_r_orig
            _ = self.template_p_orig
            _ = self.template_r
            _ = self.template_p
            _ = self.atoms_rt_map
            _ = self.atoms_pt_map
            _ = self.atoms_rt_idx_to_map
            _ = self.atoms_pt_idx_to_map
            _ = self.rt_bond_dirs_by_mapnum
            _ = self.pt_bond_dirs_by_mapnum
            _ = self.required_rt_bond_defs
            _ = self.required_bond_defs_coreatoms
            _ = self.template_has_tetra_stereo
            _ = self.template_has_doublebond_stereo
            _ = self.template_is_chiral

    def _ensure_templates(self) -> None:
        """Ensure template fragments are initialized"""
        if (
            self._template_r_orig is not None
            and self._template_p_orig is not None
            and self._template_r is not None
            and self._template_p is not None
        ):
            return

        if self._template_r_orig is None or self._template_p_orig is None:
            template_r, template_p = _get_template_frags_from_rxn(self.rxn)
            self._template_r_orig = Chem.Mol(template_r)
            self._template_p_orig = Chem.Mol(template_p)

        assert self._template_r_orig is not None
        assert self._template_p_orig is not None

        self._template_r = Chem.Mol(self._template_r_orig)
        self._template_p = Chem.Mol(self._template_p_orig)

        # Call template_atom_could_have_been_tetra to pre-assign value to atom
        for a in self._template_r.GetAtoms():
            template_atom_could_have_been_tetra(a)
        for a in self._template_p.GetAtoms():
            template_atom_could_have_been_tetra(a)

        # Precompute template invariants that depend on original atom-map numbers
        self._rt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self._template_r_orig)
        self._pt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self._template_p_orig)
        self._required_rt_bond_defs, self._required_bond_defs_coreatoms = (
            enumerate_possible_cistrans_defs(self._template_r_orig)
        )

        # Build atom-map dicts from the COPY, not the original.
        # assign_outcome_atom_mapnums mutates these atoms' map numbers in-place,
        # so they must NOT be the same objects as _template_r_orig's atoms.
        self._atoms_rt_map = {
            a.GetAtomMapNum(): a
            for a in self._template_r.GetAtoms()
            if a.GetAtomMapNum()
        }
        self._atoms_pt_map = {
            a.GetAtomMapNum(): a
            for a in self._template_p.GetAtoms()
            if a.GetAtomMapNum()
        }

        # Capture idx→mapnum from the originals while they are guaranteed clean
        self._atoms_rt_idx_to_map = {
            a.GetIdx(): a.GetAtomMapNum() for a in self._template_r_orig.GetAtoms()
        }
        self._atoms_pt_idx_to_map = {
            a.GetIdx(): a.GetAtomMapNum() for a in self._template_p_orig.GetAtoms()
        }

    @property
    def template_r_orig(self) -> Chem.Mol:
        if self._template_r_orig is None:
            self._ensure_templates()
            assert self._template_r_orig is not None
        return self._template_r_orig

    @property
    def template_p_orig(self) -> Chem.Mol:
        if self._template_p_orig is None:
            self._ensure_templates()
            assert self._template_p_orig is not None
        return self._template_p_orig

    @property
    def template_r(self) -> Chem.Mol:
        if self._template_r is None:
            self._ensure_templates()
            assert self._template_r is not None
        return self._template_r

    @property
    def template_p(self) -> Chem.Mol:
        if self._template_p is None:
            self._ensure_templates()
            assert self._template_p is not None
        return self._template_p

    @property
    def atoms_rt_map(self) -> Dict[int, Chem.Atom]:
        if self._rt_bond_dirs_by_mapnum is None:
            self._ensure_templates()
            self._rt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_r)
        if (
            self._required_rt_bond_defs is None
            or self._required_bond_defs_coreatoms is None
        ):
            self._ensure_templates()
            self._required_rt_bond_defs, self._required_bond_defs_coreatoms = (
                enumerate_possible_cistrans_defs(self.template_r)
            )
        if self._atoms_rt_map is None:
            self._ensure_templates()
            self._atoms_rt_map = {
                a.GetAtomMapNum(): a
                for a in self.template_r.GetAtoms()
                if a.GetAtomMapNum()
            }
        assert self._atoms_rt_map is not None
        return self._atoms_rt_map

    @property
    def atoms_pt_map(self) -> Dict[int, Chem.Atom]:
        if self._pt_bond_dirs_by_mapnum is None:
            self._ensure_templates()
            self._pt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_p)
        if self._atoms_pt_map is None:
            self._atoms_pt_map = {
                a.GetAtomMapNum(): a
                for a in self.template_p.GetAtoms()
                if a.GetAtomMapNum()
            }
        assert self._atoms_pt_map is not None
        return self._atoms_pt_map

    @property
    def atoms_rt_idx_to_map(self) -> Dict[int, int]:
        if self._atoms_rt_idx_to_map is None:
            self._ensure_templates()
            self._atoms_rt_idx_to_map = {
                a.GetIdx(): a.GetAtomMapNum() for a in self.template_r_orig.GetAtoms()
            }
        assert self._atoms_rt_idx_to_map is not None
        return self._atoms_rt_idx_to_map

    @property
    def atoms_pt_idx_to_map(self) -> Dict[int, int]:
        if self._atoms_pt_idx_to_map is None:
            self._ensure_templates()
            self._atoms_pt_idx_to_map = {
                a.GetIdx(): a.GetAtomMapNum() for a in self.template_p_orig.GetAtoms()
            }
        assert self._atoms_pt_idx_to_map is not None
        return self._atoms_pt_idx_to_map

    @property
    def rt_bond_dirs_by_mapnum(self) -> Dict[Tuple[int, int], BondDir]:
        if self._rt_bond_dirs_by_mapnum is None:
            self._ensure_templates()
            self._rt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_r)
        assert self._rt_bond_dirs_by_mapnum is not None
        return self._rt_bond_dirs_by_mapnum

    @property
    def pt_bond_dirs_by_mapnum(self) -> Dict[Tuple[int, int], BondDir]:
        if self._pt_bond_dirs_by_mapnum is None:
            self._ensure_templates()
            self._pt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_p)
        assert self._pt_bond_dirs_by_mapnum is not None
        return self._pt_bond_dirs_by_mapnum

    @property
    def required_rt_bond_defs(
        self,
    ) -> Dict[Tuple[int, int, int, int], Tuple[BondDir, BondDir]]:
        if self._required_rt_bond_defs is None:
            self._ensure_templates()
            self._required_rt_bond_defs, self._required_bond_defs_coreatoms = (
                enumerate_possible_cistrans_defs(self.template_r)
            )
        assert self._required_rt_bond_defs is not None
        return self._required_rt_bond_defs

    @property
    def required_bond_defs_coreatoms(self) -> Set[Tuple[int, int]]:
        if self._required_bond_defs_coreatoms is None:
            self._ensure_templates()
            self._required_rt_bond_defs, self._required_bond_defs_coreatoms = (
                enumerate_possible_cistrans_defs(self.template_r)
            )
        assert self._required_bond_defs_coreatoms is not None
        return self._required_bond_defs_coreatoms

    @property
    def template_has_tetra_stereo(self) -> bool:
        if self._template_has_tetra_stereo is None:
            self._ensure_templates()
            self._template_has_tetra_stereo = _has_tetra_stereo(
                self.template_r
            ) or _has_tetra_stereo(self.template_p)
        return self._template_has_tetra_stereo

    @property
    def template_has_doublebond_stereo(self) -> bool:
        if self._template_has_doublebond_stereo is None:
            self._ensure_templates()
            self._template_has_doublebond_stereo = _has_doublebond_stereo(
                self.template_r
            ) or _has_doublebond_stereo(self.template_p)
        return self._template_has_doublebond_stereo

    @property
    def template_is_chiral(self) -> bool:
        if self._template_is_chiral is None:
            self._ensure_templates()
            if self._template_has_tetra_stereo is None:
                self._template_has_tetra_stereo = _has_tetra_stereo(
                    self.template_r
                ) or _has_tetra_stereo(self.template_p)
            if self._template_has_doublebond_stereo is None:
                self._template_has_doublebond_stereo = _has_doublebond_stereo(
                    self.template_r
                ) or _has_doublebond_stereo(self.template_p)
            self._template_is_chiral = (
                self._template_has_tetra_stereo or self._template_has_doublebond_stereo
            )
        return self._template_is_chiral

    def reset(self) -> None:
        """Reset atom map numbers for template fragment atoms"""
        if not self._template_r or not self._template_p:
            self._ensure_templates()
        self._template_r = self._template_r_orig
        self._template_p = self._template_p_orig


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

    def __init__(
        self,
        reactant_smiles: str,
        custom_reactant_mapping: bool = False,
        lazy_init: bool = True,
    ):
        # Keep original smiles, useful for reporting
        self.reactant_smiles: str = reactant_smiles
        self.custom_mapping: bool = custom_reactant_mapping

        # Initialize into RDKit mol
        self.reactants: Chem.Mol = _initialize_reactants_from_smiles(reactant_smiles)

        self._n_atoms: Optional[int] = None
        if not self.custom_mapping:
            self._n_atoms = self.reactants.GetNumAtoms()
            for idx in range(self._n_atoms):
                self.reactants.GetAtomWithIdx(idx).SetAtomMapNum(idx + 1)

        self._n_bonds: Optional[int] = None
        self._atoms_r: Optional[Dict[int, Chem.Atom]] = None
        self._idx_to_map_num: Optional[Dict[int, int]] = None
        self._reactants_achiral: Optional[Chem.Mol] = None
        self._bonds_by_mapnum: Optional[List[Tuple[int, int, Chem.Bond]]] = None
        self._bond_dirs_by_mapnum: Optional[Dict[Tuple[int, int], BondDir]] = None
        self._atoms_across_double_bonds: Optional[
            List[Tuple[Tuple[int, int, int, int], Tuple[BondDir, BondDir], bool]]
        ] = None
        self._reactants_has_tetra_stereo: Optional[bool] = None
        self._reactants_has_doublebond_stereo: Optional[bool] = None
        self._reactants_is_chiral: Optional[bool] = None

        if not lazy_init:
            self._ensure_atom_maps()
            _ = self.atoms_r
            _ = self.reactants_achiral
            _ = self.bonds_by_mapnum
            _ = self.bond_dirs_by_mapnum
            _ = self.atoms_across_double_bonds
            _ = self.reactants_has_tetra_stereo
            _ = self.reactants_has_doublebond_stereo
            _ = self.reactants_is_chiral

    def _ensure_atom_maps(self) -> None:
        if self._atoms_r is not None and self._idx_to_map_num is not None:
            return

        atoms_r: Dict[int, Chem.Atom] = {}
        idx_to_map_num: Dict[int, int] = {}
        has_tetra_stereo = False

        if self._n_atoms is None:
            self._n_atoms = self.reactants.GetNumAtoms()
        n_atoms = self._n_atoms
        for idx in range(n_atoms):
            atom = self.reactants.GetAtomWithIdx(idx)
            map_num = atom.GetAtomMapNum()
            atoms_r[map_num] = atom
            idx_to_map_num[idx] = map_num
            if (
                not has_tetra_stereo
                and atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
            ):
                has_tetra_stereo = True

        self._atoms_r = atoms_r
        self._idx_to_map_num = idx_to_map_num
        self._reactants_has_tetra_stereo = has_tetra_stereo

    def _ensure_bond_maps(self) -> None:
        if self._bonds_by_mapnum is not None and self._bond_dirs_by_mapnum is not None:
            return

        bonds_by_mapnum: List[Tuple[int, int, Chem.Bond]] = []
        bond_dirs: Dict[Tuple[int, int], BondDir] = {}
        has_doublebond_stereo = False

        if self._n_bonds is None:
            self._n_bonds = self.reactants.GetNumBonds()
        n_bonds = self._n_bonds
        for bond_idx in range(n_bonds):
            b = self.reactants.GetBondWithIdx(bond_idx)
            i = b.GetBeginAtom().GetAtomMapNum()
            j = b.GetEndAtom().GetAtomMapNum()
            bonds_by_mapnum.append((i, j, b))

            bond_dir = b.GetBondDir()
            if bond_dir != BondDir.NONE:
                has_doublebond_stereo = True
                bond_dirs[(i, j)] = bond_dir
                bond_dirs[(j, i)] = BondDirOpposite[bond_dir]

        self._bonds_by_mapnum = bonds_by_mapnum
        self._bond_dirs_by_mapnum = bond_dirs
        self._reactants_has_doublebond_stereo = has_doublebond_stereo

    @property
    def atoms_r(self) -> Dict[int, Chem.Atom]:
        self._ensure_atom_maps()
        assert self._atoms_r is not None
        return self._atoms_r

    @property
    def reactants_achiral(self) -> Chem.Mol:
        if self._reactants_achiral is None:
            reactants_achiral = Chem.Mol(self.reactants)
            if self._n_atoms is None:
                self._n_atoms = reactants_achiral.GetNumAtoms()
            n_atoms = self._n_atoms
            if self._n_bonds is None:
                self._n_bonds = reactants_achiral.GetNumBonds()
            n_bonds = self._n_bonds
            for idx in range(n_atoms):
                reactants_achiral.GetAtomWithIdx(idx).SetChiralTag(
                    ChiralType.CHI_UNSPECIFIED
                )
            for idx in range(n_bonds):
                b = reactants_achiral.GetBondWithIdx(idx)
                b.SetStereo(BondStereo.STEREONONE)
                b.SetBondDir(BondDir.NONE)
            self._reactants_achiral = reactants_achiral
        return self._reactants_achiral

    @property
    def bonds_by_mapnum(self) -> List[Tuple[int, int, Chem.Bond]]:
        self._ensure_bond_maps()
        assert self._bonds_by_mapnum is not None
        return self._bonds_by_mapnum

    @property
    def bond_dirs_by_mapnum(self) -> Dict[Tuple[int, int], BondDir]:
        self._ensure_bond_maps()
        assert self._bond_dirs_by_mapnum is not None
        return self._bond_dirs_by_mapnum

    @property
    def atoms_across_double_bonds(
        self,
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[BondDir, BondDir], bool]]:
        if self._atoms_across_double_bonds is None:
            self._atoms_across_double_bonds = get_atoms_across_double_bonds(
                self.reactants
            )
        return self._atoms_across_double_bonds

    @property
    def reactants_has_tetra_stereo(self) -> bool:
        if self._reactants_has_tetra_stereo is None:
            self._ensure_atom_maps()
        assert self._reactants_has_tetra_stereo is not None
        return self._reactants_has_tetra_stereo

    @property
    def reactants_has_doublebond_stereo(self) -> bool:
        if self._reactants_has_doublebond_stereo is None:
            self._ensure_bond_maps()
        assert self._reactants_has_doublebond_stereo is not None
        return self._reactants_has_doublebond_stereo

    @property
    def reactants_is_chiral(self) -> bool:
        if self._reactants_is_chiral is None:
            self._reactants_is_chiral = (
                self.reactants_has_tetra_stereo or self.reactants_has_doublebond_stereo
            )
        return self._reactants_is_chiral

    def idx_to_mapnum(self, idx: int) -> int:
        self._ensure_atom_maps()
        assert self._idx_to_map_num is not None
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
    prd_maps: Set[int] = set()
    for prd in products:
        for a in prd.GetAtoms():
            if a.GetAtomMapNum():
                prd_maps.add(a.GetAtomMapNum())

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
