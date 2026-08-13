<!-- cSpell:ignore mypyc -->

# Benchmarks

All benchmarks are performed on a fresh reboot of a Latitude 5540 with an Intel Core i5-1345U and 32GB of RAM from a Linux subsystem on Windows.

## Environments

The helper script `scripts/run_speed_benchmark_envs.py` builds and runs multiple environments:

- `original`: upstream `rdchiral` installed from `git+https://github.com/connorcoley/rdchiral.git`.
- `rdchiral_plus`: this fork installed normally (pure-Python mode).
- `rdchiral_plus_mypyc`: this fork installed with `RDCHIRAL_USE_MYPYC=1`.
- `cpp`: the `rdchiral_cpp` conda-forge package (run with `--cpp`).

## Metrics

Benchmarks are executed three times and the average and standard deviation are reported.
Timings are reported as `avg (std)` in seconds.

The `*_ratio` columns are relative to **orig**, where:

- A value **> 1.0** means faster than **orig** (e.g., `3.000` is ~3x faster)
- A value **< 1.0** means slower than **orig** (e.g., `0.500` is ~2x slower)

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

- Template initialization (building `rdchiralReaction` objects).
- Reactant initialization (building `rdchiralReactants` objects).
- Template application (`rdchiralRunText`, `rdchiralRun`, `rdchiralRun_keep_mapnums`, `rdchiralRun_return_mapped_keep_mapnums`).
- Template extraction (`extract_from_reaction`).

## Benchmark 1: 1,000,000 template applications

This benchmark consists of applying 1000 templates to 1000 reactant SMILES for a total of 1,000,000 applications.

| env | reactants_init | reactants_init_ratio | templates_init | templates_init_ratio | application | application_ratio |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| orig | 0.560 (0.050) | 1.000 | 0.608 (0.035) | 1.000 | 118.250 (2.837) | 1.000 |
| rdchiral_plus | 0.551 (0.077) | 1.016 | 0.739 (0.087) | 0.822 | 39.650 (0.169) | 2.982 |
| rdchiral_plus_mypyc | 0.589 (0.025) | 0.950 | 0.733 (0.014) | 0.829 | 38.847 (1.501) | 3.044 |
| cpp | 0.163 (0.033) | 3.429 | 0.064 (0.010) | 9.500 | 44.793 (1.593) | 2.640 |

### Notes

- `reactants_init` includes loading and preprocessing the reactant set.
- `templates_init` includes loading and preprocessing the template set.
- `application` is the end-to-end runtime for performing the 1,000,000 applications.

## Benchmark 2: product-forming applications

This benchmark consists of applying 10,000 template reactant applications, all of which result in products with the original rdchiral library.

Results for this benchmark are not yet reported here.

## Benchmark 3: template extraction

This benchmark consists of extracting 10,000 templates from atom mapped reactions.

Results for this benchmark are not yet reported here.
 