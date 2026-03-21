---
description: Guidelines for writing docstrings in this codebase
---

# Docstring guidelines

## Goals

- **Clarity**: State what the function/class does.
- **Correctness**: Document the actual behavior, assumptions, and edge-cases.
- **Maintainability**: Keep docstrings updated when behavior changes.

## What to include

- **Summary**: A one-paragraph summary of what the function/class does.
- **Args**: List each parameter with type and description.
- **Returns**: Include the return type and describe the returned value(s). Use indentation for multi-line descriptions.
- **Raises**: Exceptions that can be raised and when.
- **Note**: Include any tricky behavior, external dependencies, performance characteristics, or invariants.

## Style rules

- **Format**: Use section headers exactly as `Args:`, `Returns:`, `Raises:`, `Note:`.
- **Args formatting**: One argument per line, with inline type annotation:
  - Use the format `name (Type): Description`.
- **Returns formatting**:
  - Put the return type on the same line as `Returns:` when possible.
  - If returning structured data (e.g., `Tuple[...]`, `Dict[...]`), describe components as indented sub-bullets.
- **Indentation**:
  - Indent any wrapped descriptions beneath the section they belong to.
  - Indent structured return descriptions beneath the return type.

## Example template

```python
def name_to_smiles_opsin(
    compound_name_list: List[str],
    allow_acid: bool = True,
    allow_radicals: bool = True,
    allow_bad_stereo: bool = False,
    wildcard_radicals: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Convert a list of chemical names to their corresponding SMILES representations using OPSIN.

    Args:
        compound_name_list (List[str]): A list of IUPAC or common chemical names to be converted.
        allow_acid (bool): If True, allow interpretation of acids.
        allow_radicals (bool): If True, enable radical interpretation.
        allow_bad_stereo (bool): If True, allow OPSIN to ignore uninterpretable stereochem.
        wildcard_radicals (bool): If True, output radicals as wildcards.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]:
            - First dict: Mapping from successfully converted chemical names to their SMILES strings.
            - Second dict: Mapping from chemical names that failed conversion to their error messages.

    Note:
        This function uses the `py2opsin` library to interface with OPSIN.
        Newline characters in input names are stripped to avoid CLI parsing issues.
    """
```

## When to update

- **Refactors**: If you rename parameters or change behavior, update docstrings in the same change.
- **Bug fixes**: If behavior changes, update docstrings to match the new behavior.
