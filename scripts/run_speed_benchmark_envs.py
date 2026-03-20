import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def _venv_python(venv_dir: Path) -> Path:
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
        ["uv", "pip", "install", "--python", str(venv_python), install_spec, "rdkit"],
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
        ["uv", "pip", "install", "--python", str(venv_python), ".", "--verbose"],
        env=env,
        cwd=repo_root,
    )


def _verify_import(*, python: Path) -> None:
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
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    benchmark_path = (repo_root / "scripts" / args.benchmark).resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_path}")

    venv_py = (repo_root / args.venv_py).resolve()
    venv_mypyc = (repo_root / args.venv_mypyc).resolve()
    venv_default = (repo_root / args.venv_default).resolve()
    venv_cpp = (repo_root / args.venv_cpp).resolve()

    print("=== Building pure-Python environment ===")
    _build_env(
        repo_root=repo_root, venv_dir=venv_py, use_mypyc=False, reinstall=args.reinstall
    )
    py_python = _venv_python(venv_py)
    print("--- Import verification (pure python) ---")
    _verify_import(python=py_python)
    print("--- Running benchmark (pure python) ---")
    _run_benchmark(python=py_python, repo_root=repo_root, benchmark_path=benchmark_path)

    print("\n=== Building mypyc environment ===")
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
    _run_benchmark(
        python=mypyc_python, repo_root=repo_root, benchmark_path=benchmark_path
    )

    print("\n=== Building original rdchiral environment ===\n")
    _build_env_from_url(
        install_spec="git+https://github.com/connorcoley/rdchiral.git",
        venv_dir=venv_default,
        reinstall=args.reinstall,
    )
    default_python = _venv_python(venv_default)
    print("--- Import verification (original rdchiral) ---")
    _verify_import(python=default_python)
    print("--- Running benchmark (original rdchiral) ---")
    _run_benchmark(
        python=default_python, repo_root=repo_root, benchmark_path=benchmark_path
    )

    print("\n=== Building rdchiral_cpp environment ===\n")
    _build_conda_env(env_dir=venv_cpp, reinstall=args.reinstall)
    cpp_python = _conda_python(venv_cpp)
    print("--- Running benchmark (rdchiral_cpp) ---")
    _run_benchmark(
        python=cpp_python,
        repo_root=repo_root,
        benchmark_path=benchmark_path,
        extra_args=["--cpp"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
