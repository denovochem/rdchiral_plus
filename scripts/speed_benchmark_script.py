import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

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

from rdchiral.initialization import rdchiralReactants, rdchiralReaction  # noqa: E402
from rdchiral.main import rdchiralRun, rdchiralRunText  # noqa: E402
from rdchiral.template_extractor import extract_from_reaction  # noqa: E402

RANDOM_SEED = 42
MAX_TEMPLATES = None
MAX_SMILES_INITIALIZATION_TEST = 10000
MAX_SMILES_PRE_INITIALIZED = 1000
MAX_SMILES_NOT_PRE_INITIALIZED = 100
MAX_MAPPED_REACTIONS = 100000

TEMPLATES_PATH = _data_root / "uspto_top_1k_templates.txt"
SMILES_PATH = _data_root / "zinc250k.txt"
MAPPED_REACTIONS_PATH = _data_root / "uspto_50k_mapped_reactions.txt"
SAVE_FILE_PATH = _data_root / "generated_csvs"
SAVE_FILE_PATH.mkdir(parents=True, exist_ok=True)


def load_lines(path: Path):
    return [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def write_outcomes_file(
    outcomes_path: Path, column_headers: List[str], data_to_write: List[str]
) -> None:
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
    run_rdchiralextract_time_s: float,
) -> None:
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
        timing_fh.write(f"run_rdchiralextract\t{run_rdchiralextract_time_s:.6f}\n")


def initialize_templates(
    templates: List[str], lazy_init_possible: bool = False, lazy_init: bool = False
) -> Tuple[List[Tuple[rdchiralReaction, str]], int]:
    rxn_list = []
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
    reactants_list = []
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


def shuffle_reactants_templates_order(
    rxn_list: List[Tuple[rdchiralReaction, str]],
    reactants_list: List[Tuple[rdchiralReactants, str]],
) -> List[Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]]:
    randomized_order_list = []
    for _, [rdchiral_rxn, rxn_smarts] in enumerate(rxn_list, start=1):
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
):
    outcomes = []
    for _, [_, (reactant_smi, rxn_smarts)] in enumerate(randomized_order_list, start=1):
        try:
            outcome = rdchiralRunText(rxn_smarts, reactant_smi)
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


def run_rdchiralrun(
    randomized_order_list: List[
        Tuple[Tuple[rdchiralReaction, rdchiralReactants], Tuple[str, str]]
    ],
):
    outcomes = []
    for _, [(rdchiral_reactants, rdchiral_rxn), (_, _)] in enumerate(
        randomized_order_list, start=1
    ):
        try:
            outcome = rdchiralRun(rdchiral_rxn, rdchiral_reactants)
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


def extract(reaction: str):
    split_smiles = reaction.split(">")
    reactants = split_smiles[0]
    spectators = split_smiles[1]
    products = split_smiles[2]
    reaction_id = 0
    try:
        return extract_from_reaction(
            {
                "reactants": reactants,
                "products": products,
                "spectators": spectators,
                "_id": reaction_id,
            }
        )
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        return {"reaction_id": reaction_id}


def run_rdchiralextract(mapped_reactions_list: List[str]):
    outcomes = []
    for reaction in mapped_reactions_list:
        try:
            outcome = extract(reaction)
            outcomes.append(outcome)
        except Exception:
            outcomes.append(None)
    return outcomes


_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--cpp",
    action="store_true",
    help="C++ extension mode: skip import checks and rdkit baseline",
)
_parser.add_argument(
    "--outcomes-file",
    default=None,
    help="Optional path to write per-(template,reactant) outcomes for cross-environment comparison (computed after timing)",
)
_parser.add_argument(
    "--lazy-init-possible",
    action="store_true",
    help="Enable lazy initialization of templates and reactants",
)
_parser.add_argument(
    "--save-file-prefix",
    default=None,
    help="Optional prefix for saved files",
)
_args = _parser.parse_args()
CPP_MODE = _args.cpp
LAZY_INIT_POSSIBLE = _args.lazy_init_possible
OUTCOMES_FILE = _args.outcomes_file
SAVE_FILE_PREFIX = _args.save_file_prefix


if not CPP_MODE:
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


templates = load_lines(TEMPLATES_PATH)
random.Random(RANDOM_SEED).shuffle(templates)
smiles_list = load_lines(SMILES_PATH)
random.Random(RANDOM_SEED).shuffle(smiles_list)
mapped_reactions_list = load_lines(MAPPED_REACTIONS_PATH)
random.Random(RANDOM_SEED).shuffle(mapped_reactions_list)

if MAX_TEMPLATES is not None:
    templates = templates[:MAX_TEMPLATES]
if MAX_SMILES_INITIALIZATION_TEST is not None:
    smiles_list_initialization_test = smiles_list[:MAX_SMILES_INITIALIZATION_TEST]
if MAX_SMILES_PRE_INITIALIZED is not None:
    smiles_list_pre_initialized = smiles_list[:MAX_SMILES_PRE_INITIALIZED]
if MAX_SMILES_NOT_PRE_INITIALIZED is not None:
    smiles_list_not_pre_initialized = smiles_list[:MAX_SMILES_NOT_PRE_INITIALIZED]
if MAX_MAPPED_REACTIONS is not None:
    mapped_reactions_list = mapped_reactions_list[:MAX_MAPPED_REACTIONS]


print("=== Benchmarking ===")
print("====Template initialization====")
lazy_template_init_time_s = None
lazy_reactant_init_time_s = None
eager_template_init_time_s = 0.0
eager_reactant_init_time_s = 0.0
if LAZY_INIT_POSSIBLE:
    t_start = time.perf_counter()
    _, template_init_fail = initialize_templates(
        templates, lazy_init_possible=True, lazy_init=True
    )
    t_end = time.perf_counter()
    lazy_template_init_time_s = t_end - t_start
    print(
        f"Lazy template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
    )
    print(f"Lazy template initialization failed: {template_init_fail}")
    t_start = time.perf_counter()
    _, smiles_init_fail = initialize_reactants(
        smiles_list_initialization_test, lazy_init_possible=True, lazy_init=True
    )
    t_end = time.perf_counter()
    lazy_reactant_init_time_s = t_end - t_start
    print(
        f"Lazy reactant initialization time: {t_end - t_start:.3f} seconds for {len(smiles_list_initialization_test)} reactants"
    )
    print(f"Lazy reactant initialization failed: {smiles_init_fail}")

    t_start = time.perf_counter()
    _, template_init_fail = initialize_templates(templates, lazy_init=False)
    t_end = time.perf_counter()
    eager_template_init_time_s = t_end - t_start
    print(
        f"Eager template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
    )
    print(f"Eager template initialization failed: {template_init_fail}")

    t_start = time.perf_counter()
    _, smiles_init_fail = initialize_reactants(
        smiles_list_initialization_test, lazy_init_possible=True, lazy_init=False
    )
    t_end = time.perf_counter()
    eager_reactant_init_time_s = t_end - t_start
    print(
        f"Eager reactant initialization time: {t_end - t_start:.3f} seconds for {len(smiles_list_initialization_test)} reactants"
    )
    print(f"Eager reactant initialization failed: {smiles_init_fail}")

else:
    t_start = time.perf_counter()
    _, template_init_fail = initialize_templates(
        templates, lazy_init_possible=False, lazy_init=False
    )
    t_end = time.perf_counter()
    eager_template_init_time_s = t_end - t_start
    print(
        f"Eager template initialization time: {t_end - t_start:.3f} seconds for {len(templates)} templates"
    )
    print(f"Eager template initialization failed: {template_init_fail}")

    t_start = time.perf_counter()
    _, smiles_init_fail = initialize_reactants(
        smiles_list_initialization_test, lazy_init_possible=False, lazy_init=False
    )
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
    rdchiral_reactants, rdchiral_templates
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
    SAVE_FILE_PATH / (SAVE_FILE_PREFIX + "_rdchiralRunText.csv"),
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
    rdchiral_reactants, rdchiral_templates
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
    SAVE_FILE_PATH / (SAVE_FILE_PREFIX + "_rdchiralRun.csv"),
    ["outcome"],
    outcomes_smiles,
)
print(f"run_rdchiralrun time: {t_end - t_start:.3f} seconds")


print("====rdchiralExtract====")
t_start = time.perf_counter()
outcomes = run_rdchiralextract(mapped_reactions_list)
t_end = time.perf_counter()
run_rdchiralextract_time_s = t_end - t_start
outcomes_smarts = [
    [ele.get("reaction_smarts", "")] if ele else [""] for ele in outcomes
]
write_outcomes_file(
    SAVE_FILE_PATH / (SAVE_FILE_PREFIX + "_rdchiralExtract.csv"),
    ["outcome"],
    outcomes_smarts,
)
print(f"run_rdchiralextract time: {t_end - t_start:.3f} seconds")


write_timing_file(
    SAVE_FILE_PATH / (SAVE_FILE_PREFIX + "_timings.txt"),
    lazy_template_init_time_s=lazy_template_init_time_s,
    lazy_reactant_init_time_s=lazy_reactant_init_time_s,
    eager_template_init_time_s=eager_template_init_time_s,
    eager_reactant_init_time_s=eager_reactant_init_time_s,
    run_rdchiralruntext_time_s=run_rdchiralruntext_time_s,
    run_rdchiralrun_time_s=run_rdchiralrun_time_s,
    run_rdchiralextract_time_s=run_rdchiralextract_time_s,
)
