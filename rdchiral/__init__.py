from importlib.metadata import PackageNotFoundError, version

from rdkit import RDLogger

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = [
    "rdchiralRunText",
    "rdchiralRun",
    "rdchiralReaction",
    "rdchiralReactants",
    "extract_from_reaction",
]

try:
    __version__ = version("rdchiral_plus")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
