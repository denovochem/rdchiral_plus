"""Unit tests for benchmark script utility functions."""

import random
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Add scripts/ to sys.path so we can import the benchmark script directly.
# scripts/ is not a Python package, so we use sys.path manipulation.
_repo_root = Path(__file__).resolve().parent.parent.parent
_scripts_dir = _repo_root / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from speed_benchmark_script import (  # type: ignore[import-not-found]
    _randomize_reactant_atom_mapnums,
    _serialize_rdchiralrun_return_mapped,
    load_lines,
    shuffle_reactants_templates_order,
    write_outcomes_file,
    write_timing_file,
)

# ---------------------------------------------------------------------------
# load_lines
# ---------------------------------------------------------------------------


def test_load_lines_strips_and_filters_empty(tmp_path: Path) -> None:
    """load_lines should strip whitespace and skip empty lines."""
    f = tmp_path / "test.txt"
    f.write_text("  hello  \n\n  world  \n   \n", encoding="utf-8")
    result = load_lines(f)
    assert result == ["hello", "world"]


def test_load_lines_empty_file(tmp_path: Path) -> None:
    """load_lines should return an empty list for an empty file."""
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert load_lines(f) == []


def test_load_lines_no_trailing_newline(tmp_path: Path) -> None:
    """load_lines should handle files without a trailing newline."""
    f = tmp_path / "no_newline.txt"
    f.write_text("line1\nline2", encoding="utf-8")
    assert load_lines(f) == ["line1", "line2"]


# ---------------------------------------------------------------------------
# write_outcomes_file
# ---------------------------------------------------------------------------


def test_write_outcomes_file_basic(tmp_path: Path) -> None:
    """write_outcomes_file should write tab-separated headers and rows."""
    out = tmp_path / "outcomes.csv"
    write_outcomes_file(out, ["col1", "col2"], [["a", "b"], ["c", "d"]])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "col1\tcol2"
    assert lines[1] == "a\tb"
    assert lines[2] == "c\td"


def test_write_outcomes_file_empty_row(tmp_path: Path) -> None:
    """write_outcomes_file should write blank tabs for empty rows."""
    out = tmp_path / "outcomes.csv"
    write_outcomes_file(out, ["col1", "col2"], [["a", "b"], [], ["c", "d"]])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "a\tb"
    assert lines[2] == "\t"
    assert lines[3] == "c\td"


def test_write_outcomes_file_no_data(tmp_path: Path) -> None:
    """write_outcomes_file should write only headers when data is empty."""
    out = tmp_path / "outcomes.csv"
    write_outcomes_file(out, ["col1"], [])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0] == "col1"


# ---------------------------------------------------------------------------
# write_timing_file
# ---------------------------------------------------------------------------


def test_write_timing_file_full(tmp_path: Path) -> None:
    """write_timing_file should write all timing lines when all values are provided."""
    out = tmp_path / "timings.txt"
    write_timing_file(
        out,
        lazy_template_init_time_s=1.1,
        lazy_reactant_init_time_s=2.2,
        eager_template_init_time_s=3.3,
        eager_reactant_init_time_s=4.4,
        run_rdchiralruntext_time_s=5.5,
        run_rdchiralrun_time_s=6.6,
        run_rdchiralrun_return_mapped_time_s=7.7,
        run_rdchiralrun_return_mapped_keep_mapnums_time_s=8.8,
        run_rdchiralextract_time_s=9.9,
    )
    content = out.read_text(encoding="utf-8")
    assert "lazy_template_initialization\t1.100000" in content
    assert "lazy_reactant_initialization\t2.200000" in content
    assert "eager_template_initialization\t3.300000" in content
    assert "eager_reactant_initialization\t4.400000" in content
    assert "run_rdchiralruntext\t5.500000" in content
    assert "run_rdchiralrun\t6.600000" in content
    assert "run_rdchiralrun_return_mapped\t7.700000" in content
    assert "run_rdchiralrun_return_mapped_keep_mapnums\t8.800000" in content
    assert "run_rdchiralextract\t9.900000" in content


def test_write_timing_file_no_lazy(tmp_path: Path) -> None:
    """write_timing_file should skip lazy lines when None."""
    out = tmp_path / "timings.txt"
    write_timing_file(
        out,
        lazy_template_init_time_s=None,
        lazy_reactant_init_time_s=None,
        eager_template_init_time_s=1.0,
        eager_reactant_init_time_s=2.0,
        run_rdchiralruntext_time_s=3.0,
        run_rdchiralrun_time_s=4.0,
        run_rdchiralrun_return_mapped_time_s=5.0,
        run_rdchiralrun_return_mapped_keep_mapnums_time_s=6.0,
        run_rdchiralextract_time_s=7.0,
    )
    content = out.read_text(encoding="utf-8")
    assert "lazy_template_initialization" not in content
    assert "lazy_reactant_initialization" not in content
    assert "eager_template_initialization\t1.000000" in content


# ---------------------------------------------------------------------------
# _serialize_rdchiralrun_return_mapped
# ---------------------------------------------------------------------------


def test_serialize_rdchiralrun_return_mapped_basic() -> None:
    """_serialize_rdchiralrun_return_mapped should join outcomes with |."""
    outcomes = ["CCO", "CCN"]
    mapped = {
        "CCO": ("[CH3:1][CH2:2][OH:3]", (1, 2)),
        "CCN": ("[CH3:1][CH2:2][NH2:3]", (1, 2)),
    }
    result = _serialize_rdchiralrun_return_mapped(outcomes, mapped)
    parts = result.split("|")
    assert len(parts) == 2
    assert "CCN::[CH3:1][CH2:2][NH2:3]::1,2" in parts
    assert "CCO::[CH3:1][CH2:2][OH:3]::1,2" in parts


def test_serialize_rdchiralrun_return_mapped_empty_atoms() -> None:
    """_serialize_rdchiralrun_return_mapped should handle None atoms_changed."""
    outcomes = ["CCO"]
    mapped = {"CCO": ("[CH3:1][CH2:2][OH:3]", None)}
    result = _serialize_rdchiralrun_return_mapped(outcomes, mapped)
    assert result == "CCO::[CH3:1][CH2:2][OH:3]::"


def test_serialize_rdchiralrun_return_mapped_missing_key() -> None:
    """_serialize_rdchiralrun_return_mapped should skip outcomes missing from mapped."""
    outcomes = ["CCO", "MISSING"]
    mapped = {"CCO": ("mapped_smi", (1,))}
    result = _serialize_rdchiralrun_return_mapped(outcomes, mapped)
    assert result == "CCO::mapped_smi::1"
    assert "MISSING" not in result


def test_serialize_rdchiralrun_return_mapped_empty() -> None:
    """_serialize_rdchiralrun_return_mapped should return empty string for no outcomes."""
    result = _serialize_rdchiralrun_return_mapped([], {})
    assert result == ""


# ---------------------------------------------------------------------------
# _randomize_reactant_atom_mapnums
# ---------------------------------------------------------------------------


def test_randomize_reactant_atom_mapnums_preserves_connectivity() -> None:
    """_randomize_reactant_atom_mapnums should preserve molecular connectivity."""
    from rdkit import Chem

    smiles = "CCO"
    rng = random.Random(42)
    result = _randomize_reactant_atom_mapnums(smiles, rng)
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    assert mol.GetNumAtoms() == 3


def test_randomize_reactant_atom_mapnums_deterministic() -> None:
    """_randomize_reactant_atom_mapnums should be deterministic with same seed."""
    smiles = "CCO"
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    result1 = _randomize_reactant_atom_mapnums(smiles, rng1)
    result2 = _randomize_reactant_atom_mapnums(smiles, rng2)
    assert result1 == result2


def test_randomize_reactant_atom_mapnums_different_seeds() -> None:
    """_randomize_reactant_atom_mapnums should produce different results with different seeds."""
    smiles = "CCCCO"
    rng1 = random.Random(42)
    rng2 = random.Random(99)
    result1 = _randomize_reactant_atom_mapnums(smiles, rng1)
    result2 = _randomize_reactant_atom_mapnums(smiles, rng2)
    assert result1 != result2


def test_randomize_reactant_atom_mapnums_invalid_smiles() -> None:
    """_randomize_reactant_atom_mapnums should raise ValueError for invalid SMILES."""
    rng = random.Random(42)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        _randomize_reactant_atom_mapnums("not_a_smiles", rng)


# ---------------------------------------------------------------------------
# shuffle_reactants_templates_order
# ---------------------------------------------------------------------------


def test_shuffle_reactants_templates_order_cross_product() -> None:
    """shuffle_reactants_templates_order should produce all cross-product pairs."""
    # Use mock objects since we only care about the pairing logic
    rxn_list: List[Tuple[str, str]] = [("rxn1", "smarts1"), ("rxn2", "smarts2")]
    reactants_list: List[Tuple[str, str]] = [
        ("react1", "smi1"),
        ("react2", "smi2"),
    ]
    result = shuffle_reactants_templates_order(rxn_list, reactants_list)
    # 2 templates x 2 reactants = 4 pairs
    assert len(result) == 4
    # Check all pairs exist (before shuffle, order is deterministic)
    pairs = {((r, s), (rs, sm)) for (r, s), (rs, sm) in result}
    assert (("rxn1", "react1"), ("smarts1", "smi1")) in pairs
    assert (("rxn1", "react2"), ("smarts1", "smi2")) in pairs
    assert (("rxn2", "react1"), ("smarts2", "smi1")) in pairs
    assert (("rxn2", "react2"), ("smarts2", "smi2")) in pairs


def test_shuffle_reactants_templates_order_deterministic() -> None:
    """shuffle_reactants_templates_order should be deterministic with RANDOM_SEED."""
    rxn_list: List[Tuple[str, str]] = [("rxn1", "s1"), ("rxn2", "s2"), ("rxn3", "s3")]
    reactants_list: List[Tuple[str, str]] = [("r1", "sm1"), ("r2", "sm2")]
    result1 = shuffle_reactants_templates_order(rxn_list, reactants_list)
    result2 = shuffle_reactants_templates_order(rxn_list, reactants_list)
    assert result1 == result2


def test_shuffle_reactants_templates_order_empty() -> None:
    """shuffle_reactants_templates_order should return empty list for empty inputs."""
    assert shuffle_reactants_templates_order([], []) == []
    assert shuffle_reactants_templates_order([("rxn", "s")], []) == []
    assert shuffle_reactants_templates_order([], [("r", "s")]) == []
