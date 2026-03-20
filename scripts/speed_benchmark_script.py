import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--cpp",
    action="store_true",
    help="C++ extension mode: skip import checks and rdkit baseline",
)
_args = _parser.parse_args()
CPP_MODE = _args.cpp

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun

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

# Get repository root from environment variable (set by run_speed_benchmark_envs.py)
repo_root = Path(os.environ.get("RDCHIRAL_REPO_ROOT", Path(__file__).resolve().parent))

TEMPLATES_PATH = repo_root / "uspto_top_1k_templates.txt"
SMILES_PATH = repo_root / "zinc250k.txt"

# Optional knobs
MAX_TEMPLATES = None
MAX_SMILES = 100
PRINT_EVERY = 25


def load_lines(path: Path):
    return [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


templates = load_lines(TEMPLATES_PATH)
smiles_list = load_lines(SMILES_PATH)

if MAX_TEMPLATES is not None:
    templates = templates[:MAX_TEMPLATES]
if MAX_SMILES is not None:
    smiles_list = smiles_list[:MAX_SMILES]

print(f"Loaded {len(templates)} templates, {len(smiles_list)} SMILES")
print(f"Total applications: {len(templates) * len(smiles_list):,}")

# Pre-initialize reactants once (recommended)
t0 = time.perf_counter()
reactants_list = []
bad_smiles = 0
for smi in smiles_list:
    try:
        reactants_list.append(rdchiralReactants(smi))
    except Exception:
        bad_smiles += 1
t1 = time.perf_counter()
print(f"Initialized reactants in {t1 - t0:.3f}s (bad_smiles={bad_smiles})")

t2 = time.perf_counter()
rxn_list = []
template_init_fail = 0
for smarts in templates:
    try:
        rxn_list.append(rdchiralReaction(smarts))
    except Exception:
        template_init_fail += 1
t3 = time.perf_counter()
print(
    f"Initialized templates in {t3 - t2:.3f}s (template_init_fail={template_init_fail})"
)

# Raw RDKit baseline: time only RunReactants on the achiral reactants
if not CPP_MODE:
    rdkit_total_runs = 0
    rdkit_total_outcomes = 0
    rdkit_fail = 0

    rdkit_t_start = time.perf_counter()

    for i, rxn in enumerate(rxn_list, start=1):
        for reactants in reactants_list:
            try:
                outcomes: Tuple[Any, ...] = rxn.rxn.RunReactants(
                    (reactants.reactants_achiral,)
                )
                rdkit_total_outcomes += len(outcomes)
            except Exception:
                rdkit_fail += 1
            rdkit_total_runs += 1

        if PRINT_EVERY and (i % PRINT_EVERY == 0):
            elapsed = time.perf_counter() - rdkit_t_start
            rate = rdkit_total_runs / elapsed if elapsed > 0 else float("inf")
            print(
                f"[rdkit {i}/{len(rxn_list)}] elapsed={elapsed:.2f}s runs={rdkit_total_runs:,} ({rate:,.1f} runs/s)"
            )

    rdkit_t_end = time.perf_counter()
    rdkit_elapsed = rdkit_t_end - rdkit_t_start
    rdkit_runs_per_sec = (
        rdkit_total_runs / rdkit_elapsed if rdkit_elapsed > 0 else float("inf")
    )

    print("\n=== RDKit RunReactants Results ===")
    print(f"elapsed_s: {rdkit_elapsed:.3f}")
    print(f"total_runs: {rdkit_total_runs:,}")
    print(f"runs_per_sec: {rdkit_runs_per_sec:,.2f}")
    print(f"total_outcomes: {rdkit_total_outcomes:,}")
    print(f"fail: {rdkit_fail:,}")

# Main timing loop: pre-initialize each template once, then run on all reactants
total_runs = 0
total_outcomes = 0
run_fail = 0

t_start = time.perf_counter()

for i, rxn in enumerate(rxn_list, start=1):
    for reactants in reactants_list:
        try:
            outcomes = rdchiralRun(rxn, reactants)
            total_outcomes += len(outcomes)
        except Exception:
            run_fail += 1
        total_runs += 1

    if PRINT_EVERY and (i % PRINT_EVERY == 0):
        elapsed = time.perf_counter() - t_start
        rate = total_runs / elapsed if elapsed > 0 else float("inf")
        print(
            f"[{i}/{len(rxn_list)}] elapsed={elapsed:.2f}s runs={total_runs:,} ({rate:,.1f} runs/s)"
        )

t_end = time.perf_counter()

elapsed = t_end - t_start
runs_per_sec = total_runs / elapsed if elapsed > 0 else float("inf")

print("\n=== Results ===")
print(f"elapsed_s: {elapsed:.3f}")
print(f"total_runs: {total_runs:,}")
print(f"runs_per_sec: {runs_per_sec:,.2f}")
print(f"total_outcomes: {total_outcomes:,}")
print(f"template_init_fail: {template_init_fail:,}")
print(f"run_fail: {run_fail:,}")
