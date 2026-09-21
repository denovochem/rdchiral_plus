"""Naive RDKit baseline benchmark - run via run_speed_benchmark_envs.py.

Times ``Chem.MolFromSmiles``, ``rdChemReactions.ReactionFromSmarts``, and
``ChemicalReaction.RunReactants`` without any rdchiral machinery, as a baseline
for the rdchiral benchmarks in ``speed_benchmark_script.py``. Timing keys and
output file formats intentionally match that script so downstream tooling
(``calculate_benchmark_stats.py``, ``analyze_consistency.py``) treats the
results as equivalent operations under the ``rdkit`` environment prefix.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

_script_dir = Path(__file__).resolve().parent

# `run_speed_benchmark_envs.py` sets RDCHIRAL_REPO_ROOT to <repo_root>/scripts and
# runs this script from a temporary directory. When running directly, data files
# are resolved relative to this script's location instead.
_env_root = Path(os.environ.get("RDCHIRAL_REPO_ROOT", _script_dir))
_data_root = _env_root
if (
    not (_data_root / "uspto_top_1k_templates.txt").exists()
    and (_data_root / "scripts" / "uspto_top_1k_templates.txt").exists()
):
    _data_root = _data_root / "scripts"

RANDOM_SEED = 42

TEMPLATES_PATH = _data_root / "uspto_top_1k_templates.txt"
SMILES_PATH = _data_root / "zinc250k.txt"
SAVE_FILE_PATH = _data_root / "generated_csvs"

# Type aliases for the shuffled benchmark work list. Each entry pairs the
# initialized objects with the strings they were built from, matching the
# layout produced by speed_benchmark_script.py so pair ordering is identical.
RdkitTemplateList = List[Tuple[rdChemReactions.ChemicalReaction, str]]
RdkitReactantList = List[Tuple[Chem.Mol, str]]
ShuffledPairList = List[
    Tuple[Tuple[rdChemReactions.ChemicalReaction, Chem.Mol], Tuple[str, str]]
]


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
    template_init_time_s: float,
    reactant_init_time_s: float,
    run_runreactants_text_time_s: float,
    run_runreactants_time_s: float,
) -> None:
    """
    Write benchmark timing results to a tab-separated text file.

    Timing keys intentionally reuse the rdchiral benchmark's metric names so
    that ``calculate_benchmark_stats.py`` reports the naive RDKit baseline in
    the same rows as the equivalent rdchiral operations.

    Args:
        timing_path (Path): Destination file path for the timings file.
        template_init_time_s (float): ReactionFromSmarts initialization time in
            seconds. Written as ``eager_template_initialization``.
        reactant_init_time_s (float): MolFromSmiles initialization time in
            seconds. Written as ``eager_reactant_initialization``.
        run_runreactants_text_time_s (float): Time in seconds for the combined
            ReactionFromSmarts + MolFromSmiles + RunReactants benchmark.
            Written as ``run_rdchiralruntext``.
        run_runreactants_time_s (float): Time in seconds for the pre-initialized
            RunReactants benchmark. Written as ``run_rdchiralrun``.
    """
    with timing_path.open("w", encoding="utf-8") as timing_fh:
        timing_fh.write(f"eager_template_initialization\t{template_init_time_s:.6f}\n")
        timing_fh.write(f"eager_reactant_initialization\t{reactant_init_time_s:.6f}\n")
        timing_fh.write(f"run_rdchiralruntext\t{run_runreactants_text_time_s:.6f}\n")
        timing_fh.write(f"run_rdchiralrun\t{run_runreactants_time_s:.6f}\n")


def initialize_templates_rdkit(templates: List[str]) -> Tuple[RdkitTemplateList, int]:
    """
    Initialize RDKit ChemicalReaction objects from SMARTS strings.

    This is the naive RDKit equivalent of rdchiral template initialization:
    a bare ``ReactionFromSmarts`` call with no additional preprocessing.

    Args:
        templates (List[str]): List of reaction SMARTS strings to initialize.

    Returns:
        Tuple[RdkitTemplateList, int]: A tuple containing:
            - List of (ChemicalReaction, smarts_string) tuples for successful
              initializations.
            - Count of templates that failed initialization.
    """
    rxn_list: RdkitTemplateList = []
    template_init_fail = 0
    for smarts in templates:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
        except Exception:
            template_init_fail += 1
            continue
        if rxn is None:
            template_init_fail += 1
            continue
        rxn_list.append((rxn, smarts))
    return rxn_list, template_init_fail


def initialize_reactants_rdkit(smiles_list: List[str]) -> Tuple[RdkitReactantList, int]:
    """
    Initialize RDKit Mol objects from SMILES strings.

    This is the naive RDKit equivalent of rdchiral reactant initialization:
    a bare ``MolFromSmiles`` call with no additional preprocessing.

    Args:
        smiles_list (List[str]): List of reactant SMILES strings to initialize.

    Returns:
        Tuple[RdkitReactantList, int]: A tuple containing:
            - List of (Mol, smiles_string) tuples for successful initializations.
            - Count of reactants that failed initialization.
    """
    reactants_list: RdkitReactantList = []
    reactants_init_fail = 0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            reactants_init_fail += 1
            continue
        if mol is None:
            reactants_init_fail += 1
            continue
        reactants_list.append((mol, smi))
    return reactants_list, reactants_init_fail


def shuffle_reactants_templates_order(
    rxn_list: RdkitTemplateList,
    reactants_list: RdkitReactantList,
) -> ShuffledPairList:
    """
    Create a shuffled cross-product of reaction templates and reactants.

    Generates all (template, reactant) pairs, then shuffles them deterministically
    using RANDOM_SEED to ensure reproducible benchmark ordering across environments.

    Args:
        rxn_list (RdkitTemplateList): List of (reaction, smarts) tuples from
            initialize_templates_rdkit.
        reactants_list (RdkitReactantList): List of (mol, smiles) tuples from
            initialize_reactants_rdkit.

    Returns:
        ShuffledPairList: Shuffled list of ((reaction, mol), (smarts, smiles))
            tuples.
    """
    randomized_order_list: ShuffledPairList = []
    for rxn, rxn_smarts in rxn_list:
        for mol, reactant_smi in reactants_list:
            randomized_order_list.append(((rxn, mol), (rxn_smarts, reactant_smi)))
    random.Random(RANDOM_SEED).shuffle(randomized_order_list)
    return randomized_order_list


def _outcome_to_smiles(outcome: Tuple[Chem.Mol, ...]) -> Optional[str]:
    """
    Convert a single RunReactants outcome to a canonical product SMILES string.

    Multiple product fragments are sorted and joined with "." so the result is
    directly comparable to rdchiral's merged-outcome SMILES. Atom map numbers
    are stripped first, matching rdchiralRun's default ``keep_mapnums=False``
    output. If direct SMILES generation fails (e.g., unsanitized product), a
    sanitization retry is attempted before giving up.

    Args:
        outcome (Tuple[Chem.Mol, ...]): One outcome tuple from
            ``ChemicalReaction.RunReactants``, where each element is a product
            molecule fragment.

    Returns:
        Optional[str]: Canonical product SMILES, or None if the outcome could
            not be converted.

    Note:
        This function mutates the product molecules by clearing atom map
        numbers and, on the retry path, re-sanitizing them.
    """
    for m in outcome:
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
    try:
        return ".".join(sorted(Chem.MolToSmiles(m, canonical=True) for m in outcome))
    except Exception:
        try:
            for m in outcome:
                Chem.SanitizeMol(m)
            return ".".join(
                sorted(Chem.MolToSmiles(m, canonical=True) for m in outcome)
            )
        except Exception:
            return None


def _outcomes_to_smiles_list(
    outcomes: Tuple[Tuple[Chem.Mol, ...], ...],
) -> List[str]:
    """
    Convert raw RunReactants outcomes to a sorted list of unique product SMILES.

    Args:
        outcomes (Tuple[Tuple[Chem.Mol, ...], ...]): Raw output from
            ``ChemicalReaction.RunReactants``.

    Returns:
        List[str]: Sorted, deduplicated list of canonical product SMILES.
            Outcomes that fail SMILES conversion are dropped.
    """
    smiles_set = set()
    for outcome in outcomes:
        smi = _outcome_to_smiles(outcome)
        if smi is not None:
            smiles_set.add(smi)
    return sorted(smiles_set)


def _serialize_outcome_smiles(outcome: List[str]) -> str:
    """
    Re-canonicalize a list of product SMILES into the CSV outcome format.

    Each SMILES is round-tripped through ``MolFromSmiles``/``MolToSmiles`` to
    normalize canonical ordering (including multi-fragment ordering), matching
    the serialization used by speed_benchmark_script.py. Naive RDKit products
    can occasionally yield SMILES that do not re-parse; those entries are
    dropped rather than failing the whole row.

    Args:
        outcome (List[str]): Product SMILES strings for one template
            application.

    Returns:
        str: Pipe-delimited, sorted canonical SMILES string.
    """
    canon = []
    for smi in outcome:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon.append(Chem.MolToSmiles(mol, canonical=True))
    return "|".join(sorted(canon))


def run_runreactants(
    randomized_order_list: ShuffledPairList,
) -> List[Optional[List[str]]]:
    """
    Run RDKit RunReactants on each pre-initialized (reaction, mol) pair.

    This is the naive RDKit equivalent of the rdchiralRun benchmark: product
    SMILES generation and deduplication are included so the produced output is
    comparable to rdchiralRun's return value.

    Args:
        randomized_order_list (ShuffledPairList): Shuffled list of
            ((reaction, mol), (smarts, smiles)) tuples.

    Returns:
        List[Optional[List[str]]]: List of outcomes, where each outcome is a
            sorted list of unique product SMILES strings, or None if
            RunReactants raised an exception.
    """
    outcomes: List[Optional[List[str]]] = []
    for (rxn, mol), _ in randomized_order_list:
        try:
            raw_outcomes = rxn.RunReactants((mol,))
            outcomes.append(_outcomes_to_smiles_list(raw_outcomes))
        except Exception:
            outcomes.append(None)
    return outcomes


def run_runreactants_text(
    randomized_order_list: ShuffledPairList,
) -> List[Optional[List[str]]]:
    """
    Run the full text-to-products RDKit pipeline on each (smarts, smiles) pair.

    This is the naive RDKit equivalent of the rdchiralRunText benchmark: each
    pair constructs a fresh ChemicalReaction and Mol from the input strings
    before calling RunReactants, so initialization cost is included per
    application.

    Args:
        randomized_order_list (ShuffledPairList): Shuffled list of
            ((reaction, mol), (smarts, smiles)) tuples. Only the string parts
            are used.

    Returns:
        List[Optional[List[str]]]: List of outcomes, where each outcome is a
            sorted list of unique product SMILES strings, or None if any step
            raised an exception or failed to parse.
    """
    outcomes: List[Optional[List[str]]] = []
    for _, (rxn_smarts, reactant_smi) in randomized_order_list:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
            mol = Chem.MolFromSmiles(reactant_smi)
            if rxn is None or mol is None:
                outcomes.append(None)
                continue
            raw_outcomes = rxn.RunReactants((mol,))
            outcomes.append(_outcomes_to_smiles_list(raw_outcomes))
        except Exception:
            outcomes.append(None)
    return outcomes


def main() -> None:
    """
    Run the naive RDKit baseline benchmark suite.

    Parses command-line arguments, loads benchmark data, and executes timed
    benchmarks for template initialization (ReactionFromSmarts), reactant
    initialization (MolFromSmiles), the combined text-to-products pipeline
    (equivalent to rdchiralRunText), and pre-initialized RunReactants
    (equivalent to rdchiralRun). Results are written to CSV files and a
    timings summary file using the same formats and metric keys as
    speed_benchmark_script.py.

    Data file paths, output directory, benchmark sizes, and file naming can all
    be configured via command-line arguments. Defaults resolve relative to the
    repository structure so the script works out-of-the-box when run from a
    standard clone.
    """
    parser = argparse.ArgumentParser(
        description="Run naive RDKit (RunReactants) baseline benchmarks."
    )
    parser.add_argument(
        "--save-file-prefix",
        default="rdkit",
        help="Prefix for saved files (default: rdkit)",
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
        help="Maximum pre-initialized SMILES for RunReactants benchmark (default: 1000)",
    )
    parser.add_argument(
        "--max-smiles-not-pre-initialized",
        type=int,
        default=100,
        help="Maximum non-pre-initialized SMILES for the text pipeline benchmark (default: 100)",
    )
    args = parser.parse_args()
    save_file_prefix = args.save_file_prefix
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = load_lines(args.templates_path)
    random.Random(RANDOM_SEED).shuffle(templates)
    smiles_list = load_lines(args.smiles_path)
    random.Random(RANDOM_SEED).shuffle(smiles_list)

    if args.max_templates is not None:
        templates = templates[: args.max_templates]
    smiles_list_initialization_test = smiles_list[: args.max_smiles_init_test]
    smiles_list_pre_initialized = smiles_list[: args.max_smiles_pre_initialized]
    smiles_list_not_pre_initialized = smiles_list[: args.max_smiles_not_pre_initialized]

    print("=== Benchmarking (naive RDKit baseline) ===")
    print("====Template initialization (ReactionFromSmarts)====")
    t_start = time.perf_counter()
    rdkit_templates, template_init_fail = initialize_templates_rdkit(templates)
    t_end = time.perf_counter()
    template_init_time_s = t_end - t_start
    print(
        f"Template initialization time: {template_init_time_s:.3f} seconds "
        f"for {len(templates)} templates"
    )
    print(f"Template initialization failed: {template_init_fail}")

    print("====Reactant initialization (MolFromSmiles)====")
    t_start = time.perf_counter()
    _, reactant_init_fail = initialize_reactants_rdkit(smiles_list_initialization_test)
    t_end = time.perf_counter()
    reactant_init_time_s = t_end - t_start
    print(
        f"Reactant initialization time: {reactant_init_time_s:.3f} seconds "
        f"for {len(smiles_list_initialization_test)} reactants"
    )
    print(f"Reactant initialization failed: {reactant_init_fail}")

    print("====RunReactants from text (rdchiralRunText equivalent)====")
    rdkit_reactants_text, _ = initialize_reactants_rdkit(
        smiles_list_not_pre_initialized
    )
    shuffled_pairs_text = shuffle_reactants_templates_order(
        rdkit_templates, rdkit_reactants_text
    )
    t_start = time.perf_counter()
    outcomes_text = run_runreactants_text(shuffled_pairs_text)
    t_end = time.perf_counter()
    run_runreactants_text_time_s = t_end - t_start
    outcomes_smiles = [
        [_serialize_outcome_smiles(outcome)] if outcome else [""]
        for outcome in outcomes_text
    ]
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRunText.csv"),
        ["outcome"],
        outcomes_smiles,
    )
    print(f"run_runreactants_text time: {run_runreactants_text_time_s:.3f} seconds")

    print("====RunReactants (rdchiralRun equivalent)====")
    rdkit_reactants_run, _ = initialize_reactants_rdkit(smiles_list_pre_initialized)
    shuffled_pairs_run = shuffle_reactants_templates_order(
        rdkit_templates, rdkit_reactants_run
    )
    t_start = time.perf_counter()
    outcomes_run = run_runreactants(shuffled_pairs_run)
    t_end = time.perf_counter()
    run_runreactants_time_s = t_end - t_start
    outcomes_smiles = [
        [_serialize_outcome_smiles(outcome)] if outcome else [""]
        for outcome in outcomes_run
    ]
    write_outcomes_file(
        output_dir / (save_file_prefix + "_rdchiralRun.csv"),
        ["outcome"],
        outcomes_smiles,
    )
    print(f"run_runreactants time: {run_runreactants_time_s:.3f} seconds")

    write_timing_file(
        output_dir / (save_file_prefix + "_timings.txt"),
        template_init_time_s=template_init_time_s,
        reactant_init_time_s=reactant_init_time_s,
        run_runreactants_text_time_s=run_runreactants_text_time_s,
        run_runreactants_time_s=run_runreactants_time_s,
    )


if __name__ == "__main__":
    main()
