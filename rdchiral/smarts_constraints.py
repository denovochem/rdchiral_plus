"""Post-filtering of reaction outcomes to enforce product-side SMARTS constraints.

RDKit's ``RunReactants`` ignores recursive SMARTS patterns (e.g. ``!$(C(=O)O)``)
on the **product side** of reaction SMARTS.  This module provides utilities to
extract per-atom SMARTS patterns from product templates and filter outcomes
after ``RunReactants`` returns, ensuring that product atoms satisfy the full
atom query — including recursive expressions and non-recursive constraints such
as H-count, degree, and formal charge.
"""

import re
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

# Matches the ``:N`` atom-map suffix inside a bracket atom expression, e.g.
# ``[#6&!$(C(=O)O)&+0:5]`` -> ``[#6&!$(C(=O)O)&+0]``.
_MAPNUM_RE = re.compile(r":\d+\]")


def _strip_atom_mapnum(atom_smarts: str) -> str:
    """Remove the trailing ``:N`` atom-map number from a bracket-atom SMARTS string."""
    return _MAPNUM_RE.sub("]", atom_smarts)


def extract_product_smarts_constraints(
    rxn: rdChemReactions.ChemicalReaction,
) -> List[Tuple[int, int, Chem.Mol]]:
    """
    Extract per-atom SMARTS patterns for product template atoms that carry SMARTS
    constraints beyond a bare element type.

    For each product template fragment, every mapped atom (``atom map num > 0``)
    is examined via ``MolFragmentToSmarts``.  If the resulting atom SMARTS
    contains a ``$`` (recursive SMARTS) **or** specifies constraints beyond the
    element type (e.g. ``H``, ``D``, charge, ``v``, ``x``, ``r``, ``@``), the
    pattern is kept.  Unmapped atoms (``map num == 0``) are skipped because they
    cannot be aligned to outcome atoms via ``old_mapno``.

    Args:
        rxn (rdChemReactions.ChemicalReaction): An initialised RDKit reaction
            object whose product templates will be inspected.

    Returns:
        List[Tuple[int, int, Chem.Mol]]: A list of ``(template_index,
        atom_mapnum, pattern_mol)`` tuples.  ``template_index`` is the index
        into ``rxn.GetProducts()``.  ``pattern_mol`` is a single-atom query
        molecule built from the atom SMARTS with the map number stripped.
        Returns an empty list when no product atoms carry meaningful
        constraints (the common case), enabling a zero-overhead fast path.
    """
    product_templates = rxn.GetProducts()
    patterns: List[Tuple[int, int, Chem.Mol]] = []

    for ti, pt in enumerate(product_templates):
        for a in pt.GetAtoms():
            mapnum = a.GetAtomMapNum()
            if mapnum == 0:
                continue
            try:
                atom_smarts = Chem.MolFragmentToSmarts(pt, [a.GetIdx()])
            except Exception:  # noqa: S112
                continue

            clean_smarts = _strip_atom_mapnum(atom_smarts)

            # Keep this atom if it has recursive SMARTS ($) or any constraint
            # beyond a bare element type.  A bare element looks like [#6] or
            # [C] — anything with & or ; that adds constraints is interesting.
            has_recursive = "$" in clean_smarts
            # Check for constraint operators beyond element + charge.
            # We look for & or ; that introduces H, D, v, x, r, @, a, A, #, or
            # explicit H/D counts.
            has_constraints = bool(
                re.search(r"[&;](?:H\d|D\d|v\d|x\d|r\d|@|a|A)", clean_smarts)
            )

            if not has_recursive and not has_constraints:
                continue

            patt = Chem.MolFromSmarts(clean_smarts)
            if patt is None:
                continue

            patterns.append((ti, mapnum, patt))

    return patterns


def filter_outcomes_by_smarts_constraints(
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...],
    patterns: List[Tuple[int, int, Chem.Mol]],
) -> Tuple[Tuple[Chem.Mol, ...], ...]:
    """
    Filter ``RunReactants`` outcomes by verifying product atoms satisfy their
    template SMARTS constraints.

    For each outcome and each constraint pattern, the product atom corresponding
    to the template atom is located via the ``old_mapno`` property (set by
    RDKit's ``RunReactants``).  ``GetSubstructMatches`` is then called on the
    product fragment with the single-atom pattern.  If the target atom does not
    appear in any match, the outcome is filtered out.

    This enforces **both** recursive SMARTS (which ``RunReactants`` ignores)
    and non-recursive constraints (H-count, degree, charge, etc.) that may
    mismatch after bond changes.

    Args:
        outcomes (Tuple[Tuple[Chem.Mol, ...], ...]): Raw output from
            ``ChemicalReaction.RunReactants`` — a tuple of outcomes, each
            being a tuple of product molecules.
        patterns (List[Tuple[int, int, Chem.Mol]]): Pre-computed constraint
            patterns from ``extract_product_smarts_constraints``.  Each tuple
            is ``(template_index, atom_mapnum, pattern_mol)``.

    Returns:
        Tuple[Tuple[Chem.Mol, ...], ...]: The filtered outcomes.  If
        ``patterns`` is empty, returns ``outcomes`` unchanged (zero overhead).
    """
    if not patterns:
        return outcomes

    filtered: List[Tuple[Chem.Mol, ...]] = []

    for outcome in outcomes:
        valid = True
        for ti, target_mapnum, patt in patterns:
            if ti >= len(outcome):
                continue

            product = outcome[ti]

            # Locate the product atom that originated from the template atom
            # with this map number.
            target_idx = None
            for pa in product.GetAtoms():
                if (
                    pa.HasProp("old_mapno")
                    and pa.GetIntProp("old_mapno") == target_mapnum
                ):
                    target_idx = pa.GetIdx()
                    break

            if target_idx is None:
                # Atom not found — can't verify, skip this constraint.
                continue

            matches = product.GetSubstructMatches(patt)
            if not any(target_idx in m for m in matches):
                valid = False
                break

        if valid:
            filtered.append(outcome)

    return tuple(filtered)
