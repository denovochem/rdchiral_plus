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
    """
    Find CSV files in a directory whose stem ends with the given suffix.

    Args:
        scripts_dir (Path): Directory to search for CSV files.
        suffix (str): Suffix string that the CSV file stem must end with.

    Returns:
        List[Path]: Sorted list of matching CSV file paths.
    """
    return sorted(
        p
        for p in scripts_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.stem.endswith(suffix)
    )


def _prefix_from_filename(csv_path: Path, suffix: str) -> str:
    """
    Extract the environment prefix from a CSV filename.

    Strips the suffix and any trailing underscores from the file stem to produce
    a clean prefix (e.g. "original_rdchiralRun.csv" -> "original").

    Args:
        csv_path (Path): Path to the CSV file.
        suffix (str): Suffix string that the CSV file stem ends with.

    Returns:
        str: The extracted prefix string.

    Raises:
        ValueError: If the filename does not end with the expected suffix or
            if the prefix is empty after stripping.
    """
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
    """
    Load the 'outcome' column from a tab-separated CSV file as a string Series.

    Args:
        csv_path (Path): Path to the tab-separated CSV file.

    Returns:
        pd.Series: The 'outcome' column cast to string type.

    Raises:
        KeyError: If the 'outcome' column is not present in the CSV file.
    """
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
    """
    Canonicalize reaction SMARTS strings in a Series using rdcanon.

    Args:
        outcome (pd.Series): Series of reaction SMARTS strings to canonicalize.

    Returns:
        pd.Series: Series with canonicalized SMARTS strings. Strings that fail
            canonicalization are returned unchanged.

    Raises:
        ImportError: If rdcanon is not installed.
    """
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
    """
    Build a DataFrame of outcomes from CSV files matching a suffix.

    Loads all CSV files in the directory whose stem ends with the suffix,
    extracts the outcome column from each, and combines them into a single
    DataFrame with one column per environment prefix. For rdchiralExtract
    outcomes, the SMARTS strings are canonicalized using rdcanon.

    Args:
        scripts_dir (Path): Directory containing the generated CSV files.
        suffix (str): Suffix to match CSV files by (e.g. "_rdchiralRun").

    Returns:
        pd.DataFrame: DataFrame with one column per environment, named
            "{prefix}_outcome", containing the outcome strings.

    Raises:
        FileNotFoundError: If no CSV files matching the suffix are found.
    """
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
    """
    Print counts of outcomes identical to the original rdchiral baseline.

    Compares each column in the DataFrame against the 'original_outcome' column
    and prints the count and percentage of identical values.

    Args:
        out_df (pd.DataFrame): DataFrame containing outcome columns, including
            an 'original_outcome' column as the baseline.
        label (str): Label for the comparison group, used in the printed header.

    Raises:
        KeyError: If the 'original_outcome' column is not present in the DataFrame.
    """
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
    """
    Analyze consistency of benchmark outcomes across rdchiral environments.

    Loads CSV outcome files from the specified directory, builds a
    comparison DataFrame for each operation type, and prints statistics
    showing how often each environment's outcomes match the original rdchiral
    baseline.
    """
    parser = argparse.ArgumentParser(
        description="Analyze consistency of benchmark outcomes across rdchiral environments."
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the benchmark scripts and data (default: this script's directory)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory containing generated CSV files (default: <scripts-dir>/generated_csvs)",
    )
    args = parser.parse_args()

    scripts_dir: Path = args.scripts_dir
    csv_dir: Path = (
        args.csv_dir if args.csv_dir is not None else scripts_dir / "generated_csvs"
    )

    for suffix in _SUFFIXES:
        out_df = build_outcome_dataframe(csv_dir, suffix)
        print_identical_counts_vs_original(out_df, label=suffix)


if __name__ == "__main__":
    main()
