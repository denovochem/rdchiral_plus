# mypy: ignore-errors
"""Roundtrip consistency check script - run via run_roundtrip_check.py.

This script reads atom-mapped reactions, extracts retrosynthetic templates,
applies them to the unmapped products via rdchiralRunText, and checks whether
the generated reactants match the original reactants. The count of consistent
roundtrips is printed to stdout.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional

from rdkit import Chem

from rdchiral.main import rdchiralRunText

# Try fork-specific API first, fall back to original API.
# The fork has extract_from_reaction_smiles and ExtractedTemplate;
# the original rdchiral only has extract_from_reaction and returns a plain dict.
try:
    from rdchiral.template_extractor import extract_from_reaction_smiles

    def extract(rxn: str) -> Optional[dict[str, Any]]:
        """
        Extract a template from a reaction SMILES using the fork-specific API.

        Args:
            rxn (str): Atom-mapped reaction SMILES in "reactants>>products" format.

        Returns:
            Optional[dict[str, Any]]: The extracted template dict, or None on failure.
        """
        return extract_from_reaction_smiles(rxn)
except ImportError:
    from rdchiral.template_extractor import extract_from_reaction

    def extract(rxn: str) -> Optional[dict[str, Any]]:
        """
        Extract a template from a reaction SMILES using the original rdchiral API.

        Args:
            rxn (str): Atom-mapped reaction SMILES in "reactants>>products" format.

        Returns:
            Optional[dict[str, Any]]: The extracted template dict, or None on failure.
        """
        reactants_side = rxn.split(">>")[0]
        products_side = rxn.split(">>")[1]
        return extract_from_reaction(
            {
                "reactants": reactants_side,
                "products": products_side,
                "spectators": "",
                "_id": 0,
            }
        )


_script_dir = Path(__file__).resolve().parent
_default_repo_root = _script_dir.parent
_env_root = Path(os.environ.get("RDCHIRAL_REPO_ROOT", _default_repo_root))

# Resolve data file path
if _env_root.name == "scripts" and (_env_root.parent / "rdchiral").exists():
    _data_root = _env_root
else:
    _data_root = _env_root / "scripts"

REACTIONS_PATH = _data_root / "uspto_50k_mapped_reactions.txt"


def main() -> None:
    """
    Run the roundtrip consistency check on all mapped reactions.

    Reads atom-mapped reactions from a data file, extracts templates, applies
    them to unmapped products, and counts how many roundtrips produce reactants
    consistent with the originals. The count is printed to stdout.

    The reactions file path can be overridden with ``--reactions-path``.
    """
    parser = argparse.ArgumentParser(
        description="Run roundtrip consistency check on mapped reactions."
    )
    parser.add_argument(
        "--reactions-path",
        type=Path,
        default=REACTIONS_PATH,
        help=f"Path to mapped reactions file (default: {REACTIONS_PATH})",
    )
    args = parser.parse_args()

    with open(args.reactions_path, "r") as f:
        rxns = f.read().splitlines()

    consistent = 0
    for rxn in rxns:
        try:
            product_string = rxn.split(">>")[1]
            if "." in product_string:
                continue  ## Skip products with multiple fragments
            product_mol = Chem.MolFromSmiles(product_string)
            [atom.SetAtomMapNum(0) for atom in product_mol.GetAtoms()]
            product_string_no_mapping = Chem.MolToSmiles(product_mol)

            reactant_string = rxn.split(">>")[0]
            reactant_mol = Chem.MolFromSmiles(reactant_string)
            [atom.SetAtomMapNum(0) for atom in reactant_mol.GetAtoms()]
            reactant_string_no_mapping = Chem.MolToSmiles(reactant_mol)

            out = extract(rxn)
            if out is None:
                continue
            rxn_smarts: Any = out.get("reaction_smarts", "")
            if not rxn_smarts:
                continue

            rdchiral_generated_reactants = rdchiralRunText(
                rxn_smarts, product_string_no_mapping, combine_enantiomers=False
            )

            if isinstance(rdchiral_generated_reactants, list):
                for ele in rdchiral_generated_reactants:
                    if ele in reactant_string_no_mapping:
                        consistent += 1
                        break

        except Exception:
            logging.getLogger(__name__).debug("Roundtrip check failed", exc_info=True)

    print(consistent)


if __name__ == "__main__":
    main()
