from importlib.metadata import PackageNotFoundError, version

from rdkit import RDLogger

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import (
    DEFAULT_EXTRACTED_TEMPLATE,
    extract_from_reaction,
    extract_from_reaction_smiles,
)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = [
    "rdchiralRunText",
    "rdchiralRun",
    "rdchiralReaction",
    "rdchiralReactants",
    "extract_from_reaction",
    "extract_from_reaction_smiles",
    "DEFAULT_EXTRACTED_TEMPLATE",
]

try:
    __version__ = version("rdchiral_plus")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
