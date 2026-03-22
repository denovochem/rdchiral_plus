import os
import shutil
import sys
import warnings

from setuptools import setup

# Remove buggy rdkit-stubs that ship with rdkit and cause mypyc build
# failures due to syntax errors in their .pyi files.
for _path in sys.path:
    _stubs = os.path.join(_path, "rdkit-stubs")
    if os.path.isdir(_stubs):
        shutil.rmtree(_stubs, ignore_errors=True)

# Try to use mypyc compilation
ext_modules = []
use_mypyc = os.getenv("RDCHIRAL_USE_MYPYC", "1") == "1"

if use_mypyc:
    try:
        from mypyc.build import mypycify

        mypyc_targets = [
            "rdchiral/bonds.py",
            "rdchiral/chiral.py",
            "rdchiral/clean.py",
            "rdchiral/initialization.py",
            "rdchiral/template_extractor.py",
            "rdchiral/utils.py",
            "rdchiral/main.py",
        ]

        ext_modules = mypycify(mypyc_targets)

    except Exception as e:
        warnings.warn(
            f"\n{'=' * 70}\n"
            f"WARNING: mypyc compilation failed: {e}\n"
            f"Falling back to pure Python installation (slower but functional)\n"
            f"To disable this warning, set: RDCHIRAL_USE_MYPYC=0\n"
            f"{'=' * 70}",
            UserWarning,
        )
        ext_modules = []

setup(
    ext_modules=ext_modules,
    zip_safe=False,
)
