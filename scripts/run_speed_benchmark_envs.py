"""Orchestrate speed benchmarks across multiple rdchiral environments."""

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
    if os.name == "nt":
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def _find_conda() -> str:
    """Resolve the full path to conda/mamba/micromamba.

    subprocess.run does not source ~/.bashrc, so conda may not be on PATH
    even when it works in an interactive shell.
    """
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found

    # Check common install locations
    home = Path.home()
    for candidate in (
        home / "miniforge3" / "bin" / "conda",
        home / "mambaforge" / "bin" / "conda",
        home / "miniconda3" / "bin" / "conda",
        home / "anaconda3" / "bin" / "conda",
    ):
        if candidate.exists():
            return str(candidate)

    # Last resort: ask an interactive shell
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

    # `uv` is typically installed globally, not inside the venv.
    # Use `--python` to ensure the install targets this venv.
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


def _verify_import(*, python: Path) -> None:
    """
    Verify that rdchiral can be imported from the given Python executable.

    Runs a temporary Python process that imports rdchiral.main and prints its
    file path, to confirm the correct package is installed.

    Args:
        python (Path): Path to the Python executable to test.
    """
    with tempfile.TemporaryDirectory(prefix="rdchiral_importcheck_") as d:
        tmpdir = Path(d)
        _run(
            [
                str(python),
                "-c",
                "import rdchiral.main; print(rdchiral.main.__file__)",
            ],
            cwd=tmpdir,
        )


def _run_benchmark(
    *,
    python: Path,
    repo_root: Path,
    benchmark_path: Path,
    extra_args: list[str] | None = None,
) -> None:
    """
    Run the benchmark script in an isolated temporary directory.

    Copies the benchmark script to a temp directory and runs it with the given
    Python executable, passing RDCHIRAL_REPO_ROOT as an environment variable
    so the script can locate data files. This avoids importing the in-tree
    rdchiral source instead of the installed package.

    Args:
        python (Path): Path to the Python executable to run the benchmark with.
        repo_root (Path): Path to the rdchiral repository root.
        benchmark_path (Path): Path to the benchmark script to run.
        extra_args (list[str] | None): Additional command-line arguments to pass
            to the benchmark script.
    """
    # Critical: run from a directory that does NOT contain the repo to avoid importing
    # the in-tree rdchiral/*.py instead of the installed package.
    with tempfile.TemporaryDirectory(prefix="rdchiral_bench_") as d:
        tmpdir = Path(d)
        local_benchmark = tmpdir / benchmark_path.name
        shutil.copy2(benchmark_path, local_benchmark)
        cmd = [str(python), str(local_benchmark)] + (extra_args or [])

        # Pass repository root as environment variable
        env = os.environ.copy()
        env["RDCHIRAL_REPO_ROOT"] = str(repo_root) + "/scripts"

        _run(cmd, cwd=tmpdir, env=env)


def main() -> int:
    """
    Orchestrate speed benchmarks across multiple rdchiral environments.

    Creates and manages virtual environments for pure-Python, mypyc-compiled,
    original rdchiral (from PyPI), and rdchiral_cpp (from conda-forge) builds.
    Runs the speed benchmark script in each environment and optionally
    randomizes the execution order.

    Returns:
        int: 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate speed benchmarks across multiple rdchiral environments."
    )
    parser.add_argument(
        "--benchmark",
        default="speed_benchmark_script.py",
        help="Path to the benchmark script (default: speed_benchmark_script.py)",
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
        help="Randomize the order of environment benchmarks (optionally reproducible with a seed)",
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
        "--skip-original",
        action="store_true",
        help="Skip the original rdchiral (from git) environment",
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip the rdchiral_cpp (conda-forge) environment",
    )
    parser.add_argument(
        "--original-rdchiral-spec",
        default="git+https://github.com/connorcoley/rdchiral.git",
        help="pip install spec for the original rdchiral environment (default: git+https://github.com/connorcoley/rdchiral.git)",
    )
    args = parser.parse_args()

    _check_uv()

    repo_root = Path(__file__).resolve().parent.parent
    benchmark_path = (repo_root / "scripts" / args.benchmark).resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_path}")

    venv_py = (repo_root / args.venv_py).resolve()
    venv_mypyc = (repo_root / args.venv_mypyc).resolve()
    venv_default = (repo_root / args.venv_default).resolve()
    venv_cpp = (repo_root / args.venv_cpp).resolve()

    def _env_pure_python() -> None:
        print("\n=== Building pure-python environment ===\n")
        _build_env(
            repo_root=repo_root,
            venv_dir=venv_py,
            use_mypyc=False,
            reinstall=args.reinstall,
        )
        py_python = _venv_python(venv_py)
        print("--- Import verification (pure python) ---")
        _verify_import(python=py_python)
        print("--- Running benchmark (pure python) ---")
        extra_args: list[str] = [
            "--lazy-init-possible",
            "--save-file-prefix",
            "pure_python",
        ]
        _run_benchmark(
            python=py_python,
            repo_root=repo_root,
            benchmark_path=benchmark_path,
            extra_args=extra_args,
        )

    def _env_mypyc() -> None:
        print("\n=== Building mypyc environment ===\n")
        _build_env(
            repo_root=repo_root,
            venv_dir=venv_mypyc,
            use_mypyc=True,
            reinstall=args.reinstall,
        )
        mypyc_python = _venv_python(venv_mypyc)
        print("--- Import verification (mypyc) ---")
        _verify_import(python=mypyc_python)
        print("--- Running benchmark (mypyc) ---")
        extra_args: list[str] = ["--lazy-init-possible", "--save-file-prefix", "mypyc"]
        _run_benchmark(
            python=mypyc_python,
            repo_root=repo_root,
            benchmark_path=benchmark_path,
            extra_args=extra_args,
        )

    def _env_original() -> None:
        print("\n=== Building original rdchiral environment ===\n")
        _build_env_from_url(
            install_spec=args.original_rdchiral_spec,
            venv_dir=venv_default,
            reinstall=args.reinstall,
        )
        default_python = _venv_python(venv_default)
        print("--- Import verification (original rdchiral) ---")
        _verify_import(python=default_python)
        print("--- Running benchmark (original rdchiral) ---")
        extra_args: list[str] = ["--save-file-prefix", "original"]
        _run_benchmark(
            python=default_python,
            repo_root=repo_root,
            benchmark_path=benchmark_path,
            extra_args=extra_args,
        )

    def _env_cpp() -> None:
        print("\n=== Building rdchiral_cpp environment ===\n")
        _build_conda_env(env_dir=venv_cpp, reinstall=args.reinstall)
        cpp_python = _conda_python(venv_cpp)
        print("--- Running benchmark (rdchiral_cpp) ---")
        extra_args = ["--cpp", "--save-file-prefix", "cpp"]
        _run_benchmark(
            python=cpp_python,
            repo_root=repo_root,
            benchmark_path=benchmark_path,
            extra_args=extra_args,
        )

    env_steps: list[tuple[str, Callable[[], None]]] = []
    if not args.skip_pure_python:
        env_steps.append(("pure_python", _env_pure_python))
    if not args.skip_mypyc:
        env_steps.append(("mypyc", _env_mypyc))
    if not args.skip_original:
        env_steps.append(("original", _env_original))
    if not args.skip_cpp:
        env_steps.append(("cpp", _env_cpp))

    if not env_steps:
        raise SystemExit("All environments skipped. Nothing to do.")

    random.Random(args.shuffle_seed).shuffle(env_steps)
    for _, step in env_steps:
        step()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
