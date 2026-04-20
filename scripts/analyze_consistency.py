import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from rdcanon import canon_reaction_smarts
except ImportError:  # pragma: no cover
    canon_reaction_smarts = None

_SUFFIXES: Tuple[str, ...] = (
    "_rdchiralExtract",
    "_rdchiralRun",
    "_rdchiralRunText",
    "_rdchiralRun_return_mapped",
    "_rdchiralRun_return_mapped_keep_mapnums",
)


def _find_csvs_by_suffix(scripts_dir: Path, suffix: str) -> List[Path]:
    return sorted(
        p
        for p in scripts_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.stem.endswith(suffix)
    )


def _prefix_from_filename(csv_path: Path, suffix: str) -> str:
    # Expected: <prefix><suffix>.csv (e.g. original_rdchiralRun.csv)
    stem = csv_path.stem
    if not stem.endswith(suffix):
        raise ValueError(
            f"Unexpected filename (missing suffix {suffix}): {csv_path.name}"
        )
    prefix = stem[: -len(suffix)]
    # Normalize trailing underscore, so "original_" -> "original"
    prefix = re.sub(r"_+$", "", prefix)
    if not prefix:
        raise ValueError(f"Could not parse prefix from filename: {csv_path.name}")
    return prefix


def _load_outcome_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(
        csv_path,
        skip_blank_lines=False,
        keep_default_na=False,
        na_filter=False,
        delimiter="\t",
    )
    if "outcome" not in df.columns:
        raise KeyError(f"Missing 'outcome' column in {csv_path.name}")
    return df["outcome"].astype(str)


def _canon_outcome_series(outcome: pd.Series) -> pd.Series:
    if canon_reaction_smarts is None:
        raise ImportError(
            "rdcanon is required to canonicalize reaction SMARTS. "
            "Install the 'dev' dependency group or `pip install rdcanon`."
        )

    def _canon_one(smarts: str) -> str:
        try:
            return canon_reaction_smarts(smarts)
        except Exception:
            return smarts

    return outcome.map(_canon_one)


def build_outcome_dataframe(scripts_dir: Path, suffix: str) -> pd.DataFrame:
    csv_paths = _find_csvs_by_suffix(scripts_dir, suffix)
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {scripts_dir} for suffix {suffix}")

    cols: Dict[str, pd.Series] = {}
    for csv_path in csv_paths:
        prefix = _prefix_from_filename(csv_path, suffix)
        col_name = f"{prefix}_outcome"
        outcome = _load_outcome_series(csv_path)
        if suffix == "_rdchiralExtract":
            outcome = _canon_outcome_series(outcome)
        cols[col_name] = outcome

    out_df = pd.DataFrame(cols)
    return out_df


def print_identical_counts_vs_original(out_df: pd.DataFrame, *, label: str) -> None:
    if "original_outcome" not in out_df.columns:
        raise KeyError(
            f"{label}: required column 'original_outcome' not found. Found: {list(out_df.columns)}"
        )

    original = out_df["original_outcome"]
    n = len(out_df)

    print(f"\n== {label} ==")
    print(f"rows: {n}")

    for col in out_df.columns:
        if col == "original_outcome":
            continue
        identical_mask = out_df[col].eq(original)
        identical_count = int(identical_mask.sum(skipna=False))
        pct = (identical_count / n * 100.0) if n else 0.0
        print(
            f"{col}: identical to original_outcome = {identical_count}/{n} ({pct:.2f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    scripts_dir: Path = args.scripts_dir

    for suffix in _SUFFIXES:
        out_df = build_outcome_dataframe(scripts_dir / "generated_csvs", suffix)
        print_identical_counts_vs_original(out_df, label=suffix)


if __name__ == "__main__":
    main()
