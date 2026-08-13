"""Run roundtrip consistency check across different rdchiral environments."""

import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    """
    Run a subprocess command, printing it first and raising on failure.

    Args:
        cmd (list[str]): Command and arguments to execute.
        env (dict[str, str] | None): Optional environment variables for the subprocess.
            If None, inherits the current process environment.
        cwd (Path | None): Optional working directory for the subprocess.
    """
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def _check_uv() -> None:
    """
    Verify that ``uv`` is available on PATH.

    Raises:
        SystemExit: If ``uv`` is not found, with instructions for installation.
    """
    if shutil.which("uv") is None:
        raise SystemExit(
            "'uv' is required but was not found on PATH.\n"
            "Install it with one of:\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  pip install uv\n"
            "See https://docs.astral.sh/uv/ for more information."
        )


def _venv_python(venv_dir: Path) -> Path:
    """
    Return the Python executable path for a virtual environment.

    Args:
        venv_dir (Path): Root directory of the virtual environment.

    Returns:
        Path: Path to the Python executable, accounting for platform differences.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _build_env_from_url(*, install_spec: str, venv_dir: Path, reinstall: bool) -> None:
    """Build a uv venv and install a package from a pip-compatible spec."""
    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        _run(["uv", "venv", str(venv_dir)])

    venv_python = _venv_python(venv_dir)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            install_spec,
            "rdkit",
        ],
    )


def _conda_python(env_dir: Path) -> Path:
    """
    Return the Python executable path for a conda environment.

    Args:
        env_dir (Path): Root directory of the conda environment.

    Returns:
        Path: Path to the Python executable, accounting for platform differences.
    """
    if os.name == "nt":
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def _find_conda() -> str:
    """Resolve the full path to conda/mamba/micromamba."""
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found

    home = Path.home()
    for candidate in (
        home / "miniforge3" / "bin" / "conda",
        home / "mambaforge" / "bin" / "conda",
        home / "miniconda3" / "bin" / "conda",
        home / "anaconda3" / "bin" / "conda",
    ):
        if candidate.exists():
            return str(candidate)

    try:
        result = subprocess.run(
            ["bash", "-i", "-c", "which conda"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        "Could not find conda, mamba, or micromamba. "
        "Please install one of them or ensure it is on PATH."
    )


def _build_conda_env(*, env_dir: Path, reinstall: bool) -> None:
    """Create a conda prefix env and install rdchiral_cpp from conda-forge."""
    conda = _find_conda()

    if reinstall and env_dir.exists():
        shutil.rmtree(env_dir)

    if not env_dir.exists():
        _run(
            [
                conda,
                "create",
                "--prefix",
                str(env_dir),
                "-c",
                "conda-forge",
                "rdchiral_cpp",
                "-y",
            ]
        )
    else:
        _run(
            [
                conda,
                "install",
                "--prefix",
                str(env_dir),
                "-c",
                "conda-forge",
                "rdchiral_cpp",
                "-y",
            ]
        )


def _build_env(
    *, repo_root: Path, venv_dir: Path, use_mypyc: bool, reinstall: bool
) -> None:
    """
    Build a uv virtual environment and install the local rdchiral package.

    Args:
        repo_root (Path): Path to the rdchiral repository root for installation.
        venv_dir (Path): Target directory for the virtual environment.
        use_mypyc (bool): If True, set RDCHIRAL_USE_MYPYC=1 to enable mypyc compilation.
        reinstall (bool): If True, delete and recreate the venv if it already exists.
    """
    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        _run(["uv", "venv", str(venv_dir)])

    venv_python = _venv_python(venv_dir)

    env = os.environ.copy()
    env["RDCHIRAL_USE_MYPYC"] = "1" if use_mypyc else "0"

    _run(
        ["uv", "pip", "install", "--python", str(venv_python), "."],
        env=env,
        cwd=repo_root,
    )

    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "rdkit",
        ],
        env=env,
    )


def _run_roundtrip_check(
    *,
    python: Path,
    repo_root: Path,
    check_script_path: Path,
) -> int:
    """Run the roundtrip check and return the consistent count."""
    # Critical: run from a directory that does NOT contain the repo to avoid importing
    # the in-tree rdchiral/*.py instead of the installed package.
    with tempfile.TemporaryDirectory(prefix="rdchiral_roundtrip_") as d:
        tmpdir = Path(d)
        local_script = tmpdir / check_script_path.name
        shutil.copy2(check_script_path, local_script)
        cmd = [str(python), str(local_script)]

        # Pass repository root as environment variable
        env = os.environ.copy()
        env["RDCHIRAL_REPO_ROOT"] = str(repo_root) + "/scripts"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=tmpdir,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            print(f"Error running check script: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        return int(result.stdout.strip())


def main() -> int:
    """
    Orchestrate roundtrip consistency checks across rdchiral environments.

    Creates and manages virtual environments for pure-Python, mypyc-compiled,
    default rdchiral (from PyPI), and rdchiral_cpp (from conda-forge) builds.
    Runs the roundtrip check script in each environment, collects the
    consistent counts, and prints a summary comparison.

    Returns:
        int: 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Run roundtrip consistency check across rdchiral environments"
    )
    parser.add_argument(
        "--venv-py",
        default=".venv-py",
        help="Path for the pure-Python venv (default: .venv-py)",
    )
    parser.add_argument(
        "--venv-mypyc",
        default=".venv-mypyc",
        help="Path for the mypyc venv (default: .venv-mypyc)",
    )
    parser.add_argument(
        "--venv-default",
        default=".venv-default",
        help="Path for the original rdchiral venv (default: .venv-default)",
    )
    parser.add_argument(
        "--venv-cpp",
        default=".conda-rdchiral-cpp",
        help="Path for the rdchiral_cpp conda env (default: .conda-rdchiral-cpp)",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Delete and recreate venvs before installing",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Randomize the order of environment checks",
    )
    parser.add_argument(
        "--skip-pure-python",
        action="store_true",
        help="Skip the pure-Python environment",
    )
    parser.add_argument(
        "--skip-mypyc",
        action="store_true",
        help="Skip the mypyc-compiled environment",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="Skip the default rdchiral (from PyPI) environment",
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip the rdchiral_cpp (conda-forge) environment",
    )
    parser.add_argument(
        "--original-rdchiral-spec",
        default="rdchiral",
        help="pip install spec for the default rdchiral environment (default: rdchiral from PyPI)",
    )
    args = parser.parse_args()

    _check_uv()

    repo_root = Path(__file__).resolve().parent.parent
    check_script_path = (repo_root / "scripts" / "roundtrip_check_script.py").resolve()
    if not check_script_path.exists():
        raise FileNotFoundError(f"Check script not found: {check_script_path}")

    venv_py = (repo_root / args.venv_py).resolve()
    venv_mypyc = (repo_root / args.venv_mypyc).resolve()
    venv_default = (repo_root / args.venv_default).resolve()
    venv_cpp = (repo_root / args.venv_cpp).resolve()

    results: dict[str, int] = {}

    def _check_pure_python() -> None:
        print("\n=== Pure Python ===")
        _build_env(
            repo_root=repo_root,
            venv_dir=venv_py,
            use_mypyc=False,
            reinstall=args.reinstall,
        )
        py_python = _venv_python(venv_py)
        consistent = _run_roundtrip_check(
            python=py_python,
            repo_root=repo_root,
            check_script_path=check_script_path,
        )
        results["pure_python"] = consistent
        print(f"Consistent: {consistent}")

    def _check_mypyc() -> None:
        print("\n=== MYPYC ===")
        _build_env(
            repo_root=repo_root,
            venv_dir=venv_mypyc,
            use_mypyc=True,
            reinstall=args.reinstall,
        )
        mypyc_python = _venv_python(venv_mypyc)
        consistent = _run_roundtrip_check(
            python=mypyc_python,
            repo_root=repo_root,
            check_script_path=check_script_path,
        )
        results["mypyc"] = consistent
        print(f"Consistent: {consistent}")

    def _check_default() -> None:
        print("\n=== Default (pip) ===")
        _build_env_from_url(
            install_spec=args.original_rdchiral_spec,
            venv_dir=venv_default,
            reinstall=args.reinstall,
        )
        default_python = _venv_python(venv_default)
        consistent = _run_roundtrip_check(
            python=default_python,
            repo_root=repo_root,
            check_script_path=check_script_path,
        )
        results["default"] = consistent
        print(f"Consistent: {consistent}")

    def _check_cpp() -> None:
        print("\n=== C++ (rdchiral_cpp) ===")
        _build_conda_env(
            env_dir=venv_cpp,
            reinstall=args.reinstall,
        )
        cpp_python = _conda_python(venv_cpp)
        consistent = _run_roundtrip_check(
            python=cpp_python,
            repo_root=repo_root,
            check_script_path=check_script_path,
        )
        results["cpp"] = consistent
        print(f"Consistent: {consistent}")

    checks: list[tuple[str, Callable[[], None]]] = []
    if not args.skip_pure_python:
        checks.append(("pure_python", _check_pure_python))
    if not args.skip_mypyc:
        checks.append(("mypyc", _check_mypyc))
    if not args.skip_default:
        checks.append(("default", _check_default))
    if not args.skip_cpp:
        checks.append(("cpp", _check_cpp))

    if not checks:
        raise SystemExit("All environments skipped. Nothing to do.")

    if args.shuffle_seed is not None:
        random.seed(args.shuffle_seed)
        random.shuffle(checks)

    for name, check_fn in checks:
        check_fn()

    print("\n=== Summary ===")
    for name, consistent in results.items():
        print(f"  {name}: {consistent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
