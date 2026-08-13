"""Speed benchmark script for rdchiral operations - run via run_speed_benchmark_envs.py."""

import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from rdkit import Chem

_script_dir = Path(__file__).resolve().parent
_default_repo_root = _script_dir.parent
_env_root = Path(os.environ.get("RDCHIRAL_REPO_ROOT", _default_repo_root))

# `run_speed_benchmark_envs.py` sets RDCHIRAL_REPO_ROOT to <repo_root>/scripts.
# When running directly, we want imports to come from the repo root, while data files
# should be resolved from the directory that actually contains them.
if _env_root.name == "scripts" and (_env_root.parent / "rdchiral").exists():
    repo_root = _env_root.parent
    _data_root = _env_root
else:
    repo_root = _env_root
    _data_root = _env_root

if (
    not (_data_root / "uspto_top_1k_templates.txt").exists()
    and (_data_root / "scripts" / "uspto_top_1k_templates.txt").exists()
):
    _data_root = _data_root / "scripts"

# Only add the repo root to sys.path when running standalone (no RDCHIRAL_REPO_ROOT).
# When called from run_speed_benchmark_envs.py, the venv already has rdchiral installed
# and we must NOT shadow it with the in-tree source.
if "RDCHIRAL_REPO_ROOT" not in os.environ:
    sys.path.insert(0, str(repo_root))

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction

# Type alias for extraction results. The fork's ExtractedTemplate TypedDict is
# structurally a dict; using dict[str, Any] keeps the script compatible with
# the original rdchiral which doesn't export ExtractedTemplate.
TemplateResult = dict[str, Any]

RANDOM_SEED = 42
RANDOM_REACTANT_MAP_SEED = 4242

TEMPLATES_PATH = _data_root / "uspto_top_1k_templates.txt"
SMILES_PATH = _data_root / "zinc250k.txt"
MAPPED_REACTIONS_PATH = _data_root / "uspto_50k_mapped_reactions.txt"
SAVE_FILE_PATH = _data_root / "generated_csvs"


def load_lines(path: Path) -> List[str]:
    """
    Read non-empty lines from a text file.

    Args:
        path (Path): Path to the text file to read.

    Returns:
        List[str]: Non-empty, stripped lines from the file.
    """
    return [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def write_outcomes_file(
    outcomes_path: Path, column_headers: List[str], data_to_write: List[List[str]]
) -> None:
    """
    Write benchmark outcomes to a tab-separated CSV file.

    Args:
        outcomes_path (Path): Destination file path for the outcomes CSV.
        column_headers (List[str]): Column header strings written as the first row.
        data_to_write (List[List[str]]): List of rows, where each row is a list of
            string values to be tab-joined and written.
    """
    with outcomes_path.open("w", encoding="utf-8") as outcomes_fh:
        outcomes_fh.write("\t".join(column_headers) + "\n")
        for data in data_to_write:
            if not data:
                outcomes_fh.write("\t".join([""] * len(column_headers)) + "\n")
            else:
                outcomes_fh.write("\t".join(data) + "\n")


def write_timing_file(
    timing_path: Path,
    lazy_template_init_time_s: float | None,
    lazy_reactant_init_time_s: float | None,
    eager_template_init_time_s: float,
    eager_reactant_init_time_s: float,
    run_rdchiralruntext_time_s: float,
    run_rdchiralrun_time_s: float,
    run_rdchiralrun_return_mapped_time_s: float,
    run_rdchiralrun_return_mapped_keep_mapnums_time_s: float,
    run_rdchiralextract_time_s: float,
) -> None:
    """
    Write benchmark timing results to a tab-separated text file.

    Args:
        timing_path (Path): Destination file path for the timings file.
        lazy_template_init_time_s (float | None): Lazy template initialization time
            in seconds, or None if lazy initialization was not benchmarked.
        lazy_reactant_init_time_s (float | None): Lazy reactant initialization time
            in seconds, or None if lazy initialization was not benchmarked.
        eager_template_init_time_s (float): Eager template initialization time in seconds.
        eager_reactant_init_time_s (float): Eager reactant initialization time in seconds.
        run_rdchiralruntext_time_s (float): rdchiralRunText benchmark time in seconds.
        run_rdchiralrun_time_s (float): rdchiralRun benchmark time in seconds.
        run_rdchiralrun_return_mapped_time_s (float): rdchiralRun with return_mapped
            benchmark time in seconds.
        run_rdchiralrun_return_mapped_keep_mapnums_time_s (float): rdchiralRun with
            return_mapped and keep_mapnums benchmark time in seconds.
        run_rdchiralextract_time_s (float): rdchiralExtract benchmark time in seconds.
    """
    with timing_path.open("w", encoding="utf-8") as timing_fh:
        if lazy_template_init_time_s is not None:
            timing_fh.write(
                f"lazy_template_initialization\t{lazy_template_init_time_s:.6f}\n"
            )
        if lazy_reactant_init_time_s is not None:
            timing_fh.write(
                f"lazy_reactant_initialization\t{lazy_reactant_init_time_s:.6f}\n"
            )
        timing_fh.write(
            f"eager_template_initialization\t{eager_template_init_time_s:.6f}\n"
        )
        timing_fh.write(
            f"eager_reactant_initialization\t{eager_reactant_init_time_s:.6f}\n"
        )
        timing_fh.write(f"run_rdchiralruntext\t{run_rdchiralruntext_time_s:.6f}\n")
        timing_fh.write(f"run_rdchiralrun\t{run_rdchiralrun_time_s:.6f}\n")
        timing_fh.write(
            f"run_rdchiralrun_return_mapped\t{run_rdchiralrun_return_mapped_time_s:.6f}\n"
        )
        timing_fh.write(
            f"run_rdchiralrun_return_mapped_keep_mapnums\t{run_rdchiralrun_return_mapped_keep_mapnums_time_s:.6f}\n"
        )
        timing_fh.write(f"run_rdchiralextract\t{run_rdchiralextract_time_s:.6f}\n")


def initialize_templates(
    templates: List[str], lazy_init_possible: bool = False, lazy_init: bool = False
) -> Tuple[List[Tuple[rdchiralReaction, str]], int]:
    """
    Initialize rdchiralReaction objects from SMARTS strings.

    Args:
        templates (List[str]): List of reaction SMARTS strings to initialize.
        lazy_init_possible (bool): If True, the rdchiralReaction constructor supports
            the lazy_init keyword argument. Defaults to False.
        lazy_init (bool): If True, initialize templates lazily (deferred parsing).
            Only effective when lazy_init_possible is True. Defaults to False.

    Returns:
        Tuple[List[Tuple[rdchiralReaction, str]], int]: A tuple containing:
            - List of (rdchiralReaction, smarts_string) tuples for successful initializations.
            - Count of templates that failed initialization.
    """
    rxn_list: List[Tuple[rdchiralReaction, str]] = []
    template_init_fail = 0
    for smarts in templates:
        try:
            if lazy_init_possible and lazy_init:
                rxn_list.append((rdchiralReaction(smarts, lazy_init=True), smarts))
            elif lazy_init_possible:
                rxn_list.append((rdchiralReaction(smarts, lazy_init=False), smarts))
            else:
                rxn_list.append((rdchiralReaction(smarts), smarts))
        except Exception:
            template_init_fail += 1
    return rxn_list, template_init_fail


def initialize_reactants(
    smiles_list: List[str], lazy_init_possible: bool = False, lazy_init: bool = False
) -> Tuple[List[Tuple[rdchiralReactants, str]], int]:
    """
    Initialize rdchiralReactants objects from SMILES strings.

    Args:
        smiles_list (List[str]): List of reactant SMILES strings to initialize.
        lazy_init_possible (bool): If True, the rdchiralReactants constructor supports
            the lazy_init keyword argument. Defaults to False.
        lazy_init (bool): If True, initialize reactants lazily (deferred parsing).
            Only effective when lazy_init_possible is True. Defaults to False.

    Returns:
        Tuple[List[Tuple[rdchiralReactants, str]], int]: A tuple containing:
            - List of (rdchiralReactants, smiles_string) tuples for successful initializations.
            - Count of reactants that failed initialization.
    """
    reactants_list: List[Tuple[rdchiralReactants, str]] = []
    reactants_init_fail = 0
    for smi in smiles_list:
        try:
            if lazy_init_possible and lazy_init:
                reactants_list.append((rdchiralReactants(smi, lazy_init=True), smi))
            elif lazy_init_possible:
                reactants_list.append((rdchiralReactants(smi, lazy_init=False), smi))
            else:
                reactants_list.append((rdchiralReactants(smi), smi))
        except Exception:
            reactants_init_fail += 1
    return reactants_list, reactants_init_fail


def _randomize_reactant_atom_mapnums(smiles: str, rng: random.Random) -> str:
    """
    Randomize atom map numbers in a SMILES string using the provided RNG.

    Args:
        smiles (str): Input SMILES string with atoms to remap.
        rng (random.Random): Random number generator for deterministic shuffling.

    Returns:
        str: Canonical SMILES string with randomized atom map numbers.

    Raises:
        ValueError: If the input SMILES string cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mapnums = list(range(1, mol.GetNumAtoms() + 1))
    rng.shuffle(mapnums)
    for atom, mapnum in zip(mol.GetAtoms(), mapnums):
        atom.SetAtomMapNum(mapnum)

    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def initialize_reactants_random_mapped(
    smiles_list: List[str], seed: int
) -> Tuple[List[Tuple[rdchiralReactants, str]], int]:
    """
    Initialize rdchiralReactants with randomized atom maps from SMILES strings.

    Each SMILES string has its atom map numbers randomized using a seeded RNG,
    then is initialized with custom_reactant_mapping=True to preserve the mapping.

    Args:
        smiles_list (List[str]): List of reactant SMILES strings to initialize.
        seed (int): Random seed for reproducible atom map shuffling.

    Returns:
        Tuple[List[Tuple[rdchiralReactants, str]], int]: A tuple containing:
            - List of (rdchiralReactants, mapped_smiles_string) tuples for
              successful initializations.
            - Count of reactants that failed initialization.
    """
    reactants_list: List[Tuple[rdchiralReactants, str]] = []
    reactants_init_fail = 0
    rng = random.Random(seed)
    for smi in smiles_list:
        try:
            mapped_smi = _randomize_reactant_atom_mapnums(smi, rng)
            reactants_list.append(
                (
                    rdchiralReactants(mapped_smi, custom_reactant_mapping=True),
                    mapped_smi,
                )
            )
        except Exception:
            reactants_init_fail += 1
    return reactants_list, reactants_init_fail


def shuffle_reactants_templates_order(
    rxn_list: List[Tuple[rdchiralReaction, str]],
    reactants_list: List[Tuple[rdchiralReactants, str]],
) -> List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]:
    """
    Create a shuffled cross-product of reaction templates and reactants.

    Generates all (template, reactant) pairs, then shuffles them deterministically
    using RANDOM_SEED to ensure reproducible benchmark ordering across environments.

    Args:
        rxn_list (List[Tuple[rdchiralReaction, str]]): List of (reaction, smarts)
            tuples from initialize_templates.
        reactants_list (List[Tuple[rdchiralReactants, str]]): List of
            (reactants, smiles) tuples from initialize_reactants.

    Returns:
        List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]:
            Shuffled list of ((reaction, reactants), (smarts, smiles)) tuples.
    """
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ] = []
    for rdchiral_rxn, rxn_smarts in rxn_list:
        for rdchiral_reactants, reactant_smi in reactants_list:
            randomized_order_list.append(
                ((rdchiral_rxn, rdchiral_reactants), (rxn_smarts, reactant_smi))
            )
    random.Random(RANDOM_SEED).shuffle(randomized_order_list)
    return randomized_order_list


def run_rdchiralruntext(
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ],
) -> List[Optional[List[str]]]:
    """
    Run rdchiralRunText on each (smarts, smiles) pair in the shuffled list.

    Args:
        randomized_order_list (List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]):
            Shuffled list of ((reaction, reactants), (smarts, smiles)) tuples.

    Returns:
        List[Optional[List[str]]]: List of outcomes, where each outcome is a list of
            product SMILES strings, or None if rdchiralRunText raised an exception.
    """
    outcomes: List[Optional[List[str]]] = []
    for _, (rxn_smarts, reactant_smi) in randomized_order_list:
        try:
            outcome = cast(List[str], rdchiralRunText(rxn_smarts, reactant_smi))
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


def run_rdchiralrun_return_mapped_keep_mapnums(
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ],
) -> List[Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]]:
    """
    Run rdchiralRun with return_mapped=True and keep_mapnums=True on each pair.

    Args:
        randomized_order_list (List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]):
            Shuffled list of ((reaction, reactants), (smarts, smiles)) tuples.

    Returns:
        List[Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]]:
            List of outcomes, where each outcome is a tuple of (product SMILES list,
            mapped outcomes dict), or None if rdchiralRun raised an exception.
    """
    outcomes: List[
        Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]
    ] = []
    for (rdchiral_rxn, rdchiral_reactants), _ in randomized_order_list:
        try:
            result = rdchiralRun(
                rdchiral_rxn,
                rdchiral_reactants,
                keep_mapnums=True,
                return_mapped=True,
            )
            # Original rdchiral returns [] when no outcomes; fork returns a tuple.
            if isinstance(result, list) and not result:
                outcomes.append(None)
            else:
                outcomes.append(
                    cast(
                        Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]],
                        result,
                    )
                )
        except Exception:
            outcomes.append(None)
    return outcomes


def _serialize_rdchiralrun_return_mapped(
    outcomes: List[str],
    mapped_outcomes: Dict[str, Tuple[str, Tuple[int, ...]]],
) -> str:
    """
    Serialize rdchiralRun return_mapped results into a compact string representation.

    Each outcome is serialized as "smiles::mapped_smiles::atoms_changed", and
    multiple outcomes are joined with "|".

    Args:
        outcomes (List[str]): List of product SMILES strings from rdchiralRun.
        mapped_outcomes (Dict[str, Tuple[str, Tuple[int, ...]]]): Mapping from
            product SMILES to (mapped SMILES, changed atom indices) tuples.

    Returns:
        str: Pipe-delimited serialized outcomes string.
    """
    serialized_outcomes = []
    for outcome_smiles in sorted(outcomes):
        mapped_info = mapped_outcomes.get(outcome_smiles)
        if mapped_info is None:
            continue
        mapped_smiles, atoms_changed = mapped_info
        if atoms_changed is None:
            atoms_changed_str = ""
        else:
            atoms_changed_str = ",".join(str(x) for x in atoms_changed)
        serialized_outcomes.append(
            "::".join(
                [
                    outcome_smiles,
                    "" if mapped_smiles is None else mapped_smiles,
                    atoms_changed_str,
                ]
            )
        )
    return "|".join(serialized_outcomes)


def run_rdchiralrun_return_mapped(
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ],
) -> List[Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]]:
    """
    Run rdchiralRun with return_mapped=True on each (reaction, reactants) pair.

    Args:
        randomized_order_list (List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]):
            Shuffled list of ((reaction, reactants), (smarts, smiles)) tuples.

    Returns:
        List[Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]]:
            List of outcomes, where each outcome is a tuple of (product SMILES list,
            mapped outcomes dict), or None if rdchiralRun raised an exception.
    """
    outcomes: List[
        Optional[Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]]]
    ] = []
    for (rdchiral_rxn, rdchiral_reactants), _ in randomized_order_list:
        try:
            result = rdchiralRun(rdchiral_rxn, rdchiral_reactants, return_mapped=True)
            # Original rdchiral returns [] when no outcomes; fork returns a tuple.
            if isinstance(result, list) and not result:
                outcomes.append(None)
            else:
                outcomes.append(
                    cast(
                        Tuple[List[str], Dict[str, Tuple[str, Tuple[int, ...]]]],
                        result,
                    )
                )
        except Exception:
            outcomes.append(None)
    return outcomes


def run_rdchiralrun(
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ],
) -> List[Optional[List[str]]]:
    """
    Run rdchiralRun on each (reaction, reactants) pair in the shuffled list.

    Args:
        randomized_order_list (List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]):
            Shuffled list of ((reaction, reactants), (smarts, smiles)) tuples.

    Returns:
        List[Optional[List[str]]]: List of outcomes, where each outcome is a list of
            product SMILES strings, or None if rdchiralRun raised an exception.
    """
    outcomes: List[Optional[List[str]]] = []
    for (rdchiral_rxn, rdchiral_reactants), _ in randomized_order_list:
        try:
            outcome = cast(List[str], rdchiralRun(rdchiral_rxn, rdchiral_reactants))
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


def extract(reaction: str) -> Optional[TemplateResult]:
    """
    Extract a retrosynthetic template from a mapped reaction SMILES.

    The reaction is split on ">" into reactants, spectators, and products.
    On failure, returns None instead of raising.

    Args:
        reaction (str): Atom-mapped reaction SMILES in "reactants>spectators>products" format.

    Returns:
        Optional[TemplateResult]: The extracted template, or None if extraction failed.
    """
    split_smiles = reaction.split(">")
    reactants = split_smiles[0]
    spectators = split_smiles[1]
    products = split_smiles[2]
    try:
        return cast(
            TemplateResult,
            extract_from_reaction(
                {
                    "reactants": reactants,
                    "products": products,
                    "spectators": spectators,
                    "_id": 0,
                }
            ),
        )
    except Exception:
        return None


def run_rdchiralextract(
    mapped_reactions_list: List[str],
) -> List[Optional[TemplateResult]]:
    """
    Run template extraction on a list of mapped reaction SMILES.

    Args:
        mapped_reactions_list (List[str]): List of atom-mapped reaction SMILES strings
            in "reactants>spectators>products" format.

    Returns:
        List[Optional[TemplateResult]]: List of extracted templates, or None for
            reactions where extraction failed.
    """
    outcomes: List[Optional[TemplateResult]] = []
    for reaction in mapped_reactions_list:
        try:
            outcome = extract(reaction)
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


def main() -> None:
    """
    Run the rdchiral speed benchmark suite.

    Parses command-line arguments, loads benchmark data, and executes timed
    benchmarks for template initialization, reactant initialization, rdchiralRunText,
    rdchiralRun, rdchiralRun with return_mapped, rdchiralRun with return_mapped and
    keep_mapnums, and rdchiralExtract. Results are written to CSV files and a
    timings summary file.

    Data file paths, output directory, benchmark sizes, and file naming can all
    be configured via command-line arguments. Defaults resolve relative to the
    repository structure so the script works out-of-the-box when run from a
    standard clone.
    """
    parser = argparse.ArgumentParser(
        description="Run rdchiral speed benchmarks across multiple operations."
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="C++ extension mode: skip import checks and rdkit baseline",
    )
    parser.add_argument(
        "--lazy-init-possible",
        action="store_true",
        help="Enable lazy initialization of templates and reactants",
    )
    parser.add_argument(
        "--save-file-prefix",
        default="standalone",
        help="Prefix for saved files (default: standalone)",
    )
    parser.add_argument(
        "--templates-path",
        type=Path,
        default=TEMPLATES_PATH,
        help=f"Path to templates file (default: {TEMPLATES_PATH})",
    )
    parser.add_argument(
        "--smiles-path",
        type=Path,
        default=SMILES_PATH,
        help=f"Path to SMILES file (default: {SMILES_PATH})",
    )
    parser.add_argument(
        "--mapped-reactions-path",
        type=Path,
        default=MAPPED_REACTIONS_PATH,
        help=f"Path to mapped reactions file (default: {MAPPED_REACTIONS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_FILE_PATH,
        help=f"Directory for output CSV and timing files (default: {SAVE_FILE_PATH})",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=None,
        help="Maximum number of templates to benchmark (default: all)",
    )
    parser.add_argument(
        "--max-smiles-init-test",
        type=int,
        default=10000,
        help="Maximum SMILES for initialization test (default: 10000)",
    )
    parser.add_argument(
        "--max-smiles-pre-initialized",
        type=int,
        default=1000,
        help="Maximum pre-initialized SMILES for rdchiralRun benchmarks (default: 1000)",
    )
    parser.add_argument(
        "--max-smiles-not-pre-initialized",
        type=int,
        default=100,
        help="Maximum non-pre-initialized SMILES for rdchiralRunText (default: 100)",
    )
    parser.add_argument(
        "--max-mapped-reactions",
        type=int,
        default=100000,
        help="Maximum mapped reactions for extraction benchmark (default: 100000)",
    )
    args = parser.parse_args()
    cpp_mode = args.cpp
    lazy_init_possible = args.lazy_init_possible
    save_file_prefix = args.save_file_prefix
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cpp_mode:
        print("=== Import origin checks ===")
        print("script:", __file__)
        print("sys.path[0]:", sys.path[0])

        import rdchiral
        import rdchiral.initialization
        import rdchiral.main

        print("rdchiral:", rdchiral.__file__)
        print("rdchiral.main:", rdchiral.main.__file__)
        print("rdchiral.initialization:", rdchiral.initialization.__file__)

        spec = importlib.util.find_spec("rdchiral.main")
        print("find_spec('rdchiral.main').origin:", spec.origin if spec else None)
        print("============================")

    templates = load_lines(args.templates_path)
    random.Random(RANDOM_SEED).shuffle(templates)
    smiles_list = load_lines(args.smiles_path)
    random.Random(RANDOM_SEED).shuffle(smiles_list)
    mapped_reactions_list = load_lines(args.mapped_reactions_path)
    random.Random(RANDOM_SEED).shuffle(mapped_reactions_list)

    if args.max_templates is not None:
        templates = templates[: args.max_templates]
    smiles_list_initialization_test = smiles_list[: args.max_smiles_init_test]
    smiles_list_pre_initialized = smiles_list[: args.max_smiles_pre_initialized]
    smiles_list_not_pre_initialized = smiles_list[: args.max_smiles_not_pre_initialized]
    mapped_reactions_list = mapped_reactions_list[: args.max_mapped_reactions]

    print("=== Benchmarking ===")
    print("====Template initialization====")
    lazy_template_init_time_s: Optional[float] = None
    lazy_reactant_init_time_s: Optional[float] = None
    eager_template_init_time_s = 0.0
    eager_reactant_init_time_s = 0.0
    if lazy_init_possible:
        t_start = time.perf_counter()
        template_init_fail = initialize_templates(
            templates, lazy_init_possible=True, lazy_init=True
        )[1]
        t_end = time.perf_counter()
        lazy_template_init_time_s = t_end - t_start
        print(
            f"Lazy template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
        )
        print(f"Lazy template initialization failed: {template_init_fail}")
        t_start = time.perf_counter()
        smiles_init_fail = initialize_reactants(
            smiles_list_initialization_test, lazy_init_possible=True, lazy_init=True
        )[1]
        t_end = time.perf_counter()
        lazy_reactant_init_time_s = t_end - t_start
        print(
            f"Lazy reactant initialization time: {t_end - t_start:.3f} seconds for {len(smiles_list_initialization_test)} reactants"
        )
        print(f"Lazy reactant initialization failed: {smiles_init_fail}")

        t_start = time.perf_counter()
        template_init_fail = initialize_templates(templates, lazy_init=False)[1]
        t_end = time.perf_counter()
        eager_template_init_time_s = t_end - t_start
        print(
            f"Eager template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
        )
        print(f"Eager template initialization failed: {template_init_fail}")

        t_start = time.perf_counter()
        smiles_init_fail = initialize_reactants(
            smiles_list_initialization_test, lazy_init_possible=True, lazy_init=False
        )[1]
        t_end = time.perf_counter()
        eager_reactant_init_time_s = t_end - t_start
        print(
            f"Eager reactant initialization time: {t_end - t_start:.3f} seconds for {len(smiles_list_initialization_test)} reactants"
        )
        print(f"Eager reactant initialization failed: {smiles_init_fail}")

    else:
        t_start = time.perf_counter()
        template_init_fail = initialize_templates(
            templates, lazy_init_possible=False, lazy_init=False
        )[1]
        t_end = time.perf_counter()
        eager_template_init_time_s = t_end - t_start
        print(
            f"Eager template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
        )
        print(f"Eager template initialization failed: {template_init_fail}")

        t_start = time.perf_counter()
        smiles_init_fail = initialize_reactants(
            smiles_list_initialization_test, lazy_init_possible=False, lazy_init=False
        )[1]
        t_end = time.perf_counter()
        eager_reactant_init_time_s = t_end - t_start
        print(
            f"Eager reactant initialization time: {t_end - t_start:.3f} seconds for {len(smiles_list_initialization_test)} reactants"
        )
        print(f"Eager reactant initialization failed: {smiles_init_fail}")

    print("====rdchiralRunText====")
    rdchiral_templates, template_init_fail = initialize_templates(
        templates, lazy_init=False
    )
    rdchiral_reactants, smiles_init_fail = initialize_reactants(
        smiles_list_not_pre_initialized, lazy_init=False
    )

    shuffled_smiles_list_not_pre_initialized = shuffle_reactants_templates_order(
        rdchiral_templates, rdchiral_reactants
    )
    t_start = time.perf_counter()
    outcomes = run_rdchiralruntext(shuffled_smiles_list_not_pre_initialized)
    t_end = time.perf_counter()
    run_rdchiralruntext_time_s = t_end - t_start
    outcomes_smiles = [
        ["|".join(sorted([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in outcome]))]
        if outcome
        else [""]
        for outcome in outcomes
    ]
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRunText.csv"),
        ["outcome"],
        outcomes_smiles,
    )
    print(f"run_rdchiralruntext time: {t_end - t_start:.3f} seconds")

    print("====rdchiralRun====")
    rdchiral_templates, template_init_fail = initialize_templates(
        templates, lazy_init=False
    )
    rdchiral_reactants, smiles_init_fail = initialize_reactants(
        smiles_list_pre_initialized, lazy_init=False
    )

    shuffled_smiles_list_pre_initialized = shuffle_reactants_templates_order(
        rdchiral_templates, rdchiral_reactants
    )
    t_start = time.perf_counter()
    outcomes = run_rdchiralrun(shuffled_smiles_list_pre_initialized)
    t_end = time.perf_counter()
    run_rdchiralrun_time_s = t_end - t_start
    outcomes_smiles = [
        ["|".join(sorted([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in outcome]))]
        if outcome
        else [""]
        for outcome in outcomes
    ]
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRun.csv"),
        ["outcome"],
        outcomes_smiles,
    )
    print(f"run_rdchiralrun time: {t_end - t_start:.3f} seconds")

    print("====rdchiralRun (return_mapped=True)====")
    rdchiral_templates, template_init_fail = initialize_templates(
        templates, lazy_init=False
    )
    rdchiral_reactants, smiles_init_fail = initialize_reactants(
        smiles_list_pre_initialized, lazy_init=False
    )

    shuffled_smiles_list_pre_initialized = shuffle_reactants_templates_order(
        rdchiral_templates, rdchiral_reactants
    )
    t_start = time.perf_counter()
    outcomes_mapped = run_rdchiralrun_return_mapped(
        shuffled_smiles_list_pre_initialized
    )
    t_end = time.perf_counter()
    run_rdchiralrun_return_mapped_time_s = t_end - t_start
    outcomes_serialized = []
    for ele in outcomes_mapped:
        if ele is None:
            outcomes_serialized.append([""])
            continue
        outcome, mapped_outcomes = ele
        outcomes_serialized.append(
            [_serialize_rdchiralrun_return_mapped(outcome, mapped_outcomes)]
        )
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRun_return_mapped.csv"),
        ["outcome"],
        outcomes_serialized,
    )
    print(f"run_rdchiralrun (return_mapped=True) time: {t_end - t_start:.3f} seconds")

    print(
        "====rdchiralRun (return_mapped=True, keep_mapnums=True, seeded reactant maps)===="
    )
    rdchiral_templates, template_init_fail = initialize_templates(
        templates, lazy_init=False
    )
    rdchiral_reactants, smiles_init_fail = initialize_reactants_random_mapped(
        smiles_list_pre_initialized, seed=RANDOM_REACTANT_MAP_SEED
    )

    shuffled_smiles_list_pre_initialized = shuffle_reactants_templates_order(
        rdchiral_templates, rdchiral_reactants
    )
    t_start = time.perf_counter()
    outcomes_keep_mapnums = run_rdchiralrun_return_mapped_keep_mapnums(
        shuffled_smiles_list_pre_initialized
    )
    t_end = time.perf_counter()
    run_rdchiralrun_return_mapped_keep_mapnums_time_s = t_end - t_start
    outcomes_serialized = []
    for ele in outcomes_keep_mapnums:
        if ele is None:
            outcomes_serialized.append([""])
            continue
        outcome, mapped_outcomes = ele
        outcomes_serialized.append(
            [_serialize_rdchiralrun_return_mapped(outcome, mapped_outcomes)]
        )
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRun_return_mapped_keep_mapnums.csv"),
        ["outcome"],
        outcomes_serialized,
    )
    print(
        "run_rdchiralrun (return_mapped=True, keep_mapnums=True) time: "
        f"{t_end - t_start:.3f} seconds"
    )

    print("====rdchiralExtract====")
    t_start = time.perf_counter()
    outcomes_extract = run_rdchiralextract(mapped_reactions_list)
    t_end = time.perf_counter()
    run_rdchiralextract_time_s = t_end - t_start
    outcomes_smarts = [
        [ele.get("reaction_smarts", "")] if ele else [""] for ele in outcomes_extract
    ]
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralExtract.csv"),
        ["outcome"],
        outcomes_smarts,
    )
    print(f"run_rdchiralextract time: {t_end - t_start:.3f} seconds")

    write_timing_file(
        output_dir / (save_file_prefix + "_timings.txt"),
        lazy_template_init_time_s=lazy_template_init_time_s,
        lazy_reactant_init_time_s=lazy_reactant_init_time_s,
        eager_template_init_time_s=eager_template_init_time_s,
        eager_reactant_init_time_s=eager_reactant_init_time_s,
        run_rdchiralruntext_time_s=run_rdchiralruntext_time_s,
        run_rdchiralrun_time_s=run_rdchiralrun_time_s,
        run_rdchiralrun_return_mapped_time_s=run_rdchiralrun_return_mapped_time_s,
        run_rdchiralrun_return_mapped_keep_mapnums_time_s=run_rdchiralrun_return_mapped_keep_mapnums_time_s,
        run_rdchiralextract_time_s=run_rdchiralextract_time_s,
    )


if __name__ == "__main__":
    main()
