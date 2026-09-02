<!-- cSpell:ignore mypyc -->

# Benchmarks

All benchmarks are performed on a fresh reboot of a Latitude 5540 with an Intel Core i5-1345U and 32GB of RAM from a Linux subsystem on Windows.

## Environments

The helper script `scripts/run_speed_benchmark_envs.py` builds and runs multiple environments:

- `RDChiral`: upstream `RDChiral` installed from `git+https://github.com/connorcoley/rdchiral.git`.
- `rdchiral_plus`: this fork installed normally (pure-Python mode).
- `rdchiral_plus_mypyc`: this fork installed with `RDCHIRAL_USE_MYPYC=1`.
- `rdchiral_cpp`: the `rdchiral_cpp` conda-forge package (run with `--cpp`).

## Metrics

Benchmarks are executed three times and the average and standard deviation are reported.
Timings are reported as `avg (std)` in seconds.

The `*_ratio` columns are relative to **RDChiral**, where:

- A value **> 1.0** means faster than **RDChiral** (e.g., `3.000` is ~3x faster)
- A value **< 1.0** means slower than **RDChiral** (e.g., `0.500` is ~2x slower)

## Reproducing

The benchmark runner builds isolated environments and executes the selected benchmark script outside the repo directory to avoid accidentally importing the in-tree sources.

```bash
python scripts/run_speed_benchmark_envs.py --reinstall
```

## Benchmark methodology

### Runner behavior

Benchmarks are orchestrated by `scripts/run_speed_benchmark_envs.py`.

- Each environment is installed into an isolated env (uv venvs for `orig`/`rdchiral_plus`/`rdchiral_plus_mypyc`, and a conda prefix env for `cpp`).
- The benchmark script is copied to a temporary directory and executed from there to avoid importing in-tree sources.
- The runner sets `RDCHIRAL_REPO_ROOT` so the benchmark script can find the repository data files.

### Workload inputs and determinism

The default benchmark script is `scripts/speed_benchmark_script.py`.

- Templates are loaded from `uspto_top_1k_templates.txt`.
- Reactant SMILES are loaded from `zinc250k.txt`.
- Atom-mapped reactions are loaded from `scripts/uspto_50k_mapped_reactions.txt`.
- The script shuffles inputs deterministically with `RANDOM_SEED = 42`.

### What is measured

The script reports timings for:

- Template initialization (building `rdchiralReaction` objects from 1000 templates).
- Reactant initialization (building `rdchiralReactants` objects from 10000 reactant SMILES).
- Template application via `rdchiralRunText` (100 templates x 100 SMILES = 10,000 applications).
- Template application via `rdchiralRun` (1000 templates x 1000 SMILES = 1,000,000 applications).
- Template application via `rdchiralRun` with `return_mapped=True` (1,000,000 applications).
- Template application via `rdchiralRun` with `return_mapped=True, keep_mapnums=True` (1,000,000 applications).
- Template extraction via `extract_from_reaction` (50,016 mapped reactions).

## Benchmark 1: Template initialization

Building `rdchiralReaction` objects from 1000 templates.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 0.659 (0.084) | 1.000 |
| rdchiral_plus | 0.022 (0.006) | 29.631 |
| rdchiral_plus_mypyc | 0.029 (0.003) | 22.892 |
| rdchiral_cpp | 0.157 (0.052) | 4.190 |

## Benchmark 2: Reactant initialization

Building `rdchiralReactants` objects from 10000 reactant SMILES.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 6.543 (0.476) | 1.000 |
| rdchiral_plus | 4.252 (0.530) | 1.539 |
| rdchiral_plus_mypyc | 4.458 (0.044) | 1.468 |
| rdchiral_cpp | 1.896 (0.169) | 3.451 |

## Benchmark 3: rdchiralRunText

Applying 100 templates to 100 reactant SMILES via `rdchiralRunText` for a total of 10,000 applications.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 134.471 (6.558) | 1.000 |
| rdchiral_plus | 26.538 (5.134) | 5.067 |
| rdchiral_plus_mypyc | 27.198 (1.685) | 4.944 |
| rdchiral_cpp | 30.906 (0.908) | 4.351 |

## Benchmark 4: rdchiralRun

Applying 1000 templates to 1000 reactant SMILES via `rdchiralRun` for a total of 1,000,000 applications.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 129.738 (0.600) | 1.000 |
| rdchiral_plus | 71.490 (5.881) | 1.815 |
| rdchiral_plus_mypyc | 72.052 (2.138) | 1.801 |
| rdchiral_cpp | 52.891 (0.677) | 2.453 |

## Benchmark 5: rdchiralRun with return_mapped=True

Applying 1000 templates to 1000 reactant SMILES via `rdchiralRun` with `return_mapped=True` for a total of 1,000,000 applications.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 132.008 (2.064) | 1.000 |
| rdchiral_plus | 73.715 (1.796) | 1.791 |
| rdchiral_plus_mypyc | 75.263 (2.950) | 1.754 |
| rdchiral_cpp | 54.914 (0.436) | 2.404 |

## Benchmark 6: rdchiralRun with return_mapped=True, keep_mapnums=True

Applying 1000 templates to 1000 reactant SMILES via `rdchiralRun` with `return_mapped=True` and `keep_mapnums=True` for a total of 1,000,000 applications.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 121.588 (0.332) | 1.000 |
| rdchiral_plus | 68.792 (2.039) | 1.767 |
| rdchiral_plus_mypyc | 69.108 (2.267) | 1.759 |
| rdchiral_cpp | not supported | — |

## Benchmark 7: Template extraction

Extracting templates from 50,016 atom-mapped reactions via `extract_from_reaction`.

| env | time (s) | ratio |
| --- | :---: | :---: |
| RDChiral | 268.668 (1.878) | 1.000 |
| rdchiral_plus | 153.925 (12.242) | 1.745 |
| rdchiral_plus_mypyc | 141.834 (3.408) | 1.894 |
| rdchiral_cpp | 86.310 (5.668) | 3.113 |