import pytest

from rdchiral.clean import canonicalize_outcome_smiles, combine_enantiomers_into_racemic


def test_canonicalize_outcome_smiles_invalid_raises_when_ensure_true():
    with pytest.raises(ValueError, match=r"Invalid SMILES"):
        canonicalize_outcome_smiles("this_is_not_smiles", ensure=True)


def test_canonicalize_outcome_smiles_invalid_allowed_when_ensure_false_sorts_fragments():
    assert canonicalize_outcome_smiles("b.a", ensure=False) == "a.b"


def test_canonicalize_outcome_smiles_sorts_fragments_after_rdkit_roundtrip():
    assert canonicalize_outcome_smiles("O.C", ensure=True) == "C.O"


def test_combine_enantiomers_into_racemic_collapses_single_tetrahedral_center_pair():
    outcomes = {"C[C@H](O)F", "C[C@@H](O)F"}

    result = combine_enantiomers_into_racemic(outcomes)

    assert result is outcomes
    assert result == {"CC(O)F"}


def test_combine_enantiomers_into_racemic_collapses_simple_alkene_slash_pair():
    outcomes = {"F/C=C/F", "F/C=C\\F"}

    result = combine_enantiomers_into_racemic(outcomes)

    assert result is outcomes
    assert len(result) == 1
    assert next(iter(result)) == canonicalize_outcome_smiles("FC=CF", ensure=True)
