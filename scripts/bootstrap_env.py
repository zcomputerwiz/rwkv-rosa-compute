#!/usr/bin/env python3
"""Bootstrap a working development environment for rosa-compute.

Runs BEFORE the project is installed, so this module must stay pure standard
library. Do not import torch, numpy, rosa_compute, or exp0 at module level.

What it does, in order:
  1. Verify the interpreter is Python 3.10+.
  2. Initialize the pinned git submodules (external/RWKV-LM, external/rosa_soft).
  3. Create .venv (skipped if already inside a virtual environment).
  4. Install torch from a platform-appropriate index, then requirements-dev.txt,
     then the project in editable mode.
  5. Verify the result: torch version, CUDA availability, device name, imports.
  6. Optionally persist environment variables into the venv activation scripts.

Typical use:

    python scripts/bootstrap_env.py
    python scripts/bootstrap_env.py --cpu
    python scripts/bootstrap_env.py --check
    python scripts/bootstrap_env.py --persist-env ROSA_MODEL_PATH=D:/models/rosa.pth

This script cannot activate the venv for you -- a child process cannot modify
its parent shell. The activation command is printed at the end.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"

# Verified to carry current Windows wheels. Override with --torch-index if your
# driver needs a different CUDA build; see https://pytorch.org/get-started/locally/
DEFAULT_WINDOWS_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

SUBMODULE_MARKERS = {
    "external/RWKV-LM": "RWKV-v7/rwkv_v7_numpy.py",
    "external/rosa_soft": "pyproject.toml",
}

BLOCK_START = "# >>> rosa-compute bootstrap >>>"
BLOCK_END = "# <<< rosa-compute bootstrap <<<"


class BootstrapError(RuntimeError):
    pass


def info(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    info("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise BootstrapError(f"command failed with exit code {result.returncode}")


def capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BootstrapError(
            f"command failed: {' '.join(str(c) for c in cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        raise BootstrapError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(pyproject requires-python), found {platform.python_version()}"
        )
    info(f"Python {platform.python_version()} on {platform.system()} - OK")


def ensure_submodules() -> None:
    missing = [
        path
        for path, marker in SUBMODULE_MARKERS.items()
        if not (REPO_ROOT / path / marker).exists()
    ]
    if not missing:
        info("Submodules already initialized")
        return
    if shutil.which("git") is None:
        raise BootstrapError(
            "git not found on PATH, and these submodules are missing: "
            + ", ".join(missing)
        )
    info(f"Initializing submodules: {', '.join(missing)}")
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_ROOT)
    still_missing = [
        path
        for path, marker in SUBMODULE_MARKERS.items()
        if not (REPO_ROOT / path / marker).exists()
    ]
    if still_missing:
        raise BootstrapError(
            "Submodules still missing after init: " + ", ".join(still_missing)
        )


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(use_venv: bool) -> Path:
    if not use_venv:
        info("--no-venv given; installing into the current interpreter")
        return Path(sys.executable)
    if in_virtualenv():
        info(f"Already inside a virtual environment ({sys.prefix}); using it")
        return Path(sys.executable)
    py = venv_python(VENV_DIR)
    if py.exists():
        info(f"Reusing existing venv at {VENV_DIR}")
        return py
    info(f"Creating venv at {VENV_DIR}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not py.exists():
        raise BootstrapError(f"venv creation did not produce {py}")
    return py


def resolve_torch_index(args: argparse.Namespace) -> str | None:
    """Return the --index-url for torch, or None to use default PyPI."""
    if args.torch_index:
        return args.torch_index
    if args.cpu:
        return CPU_INDEX
    if platform.system() == "Windows":
        # PyPI's default Windows wheel is CPU-only; an explicit CUDA index is
        # required or the GPU is silently unused.
        return DEFAULT_WINDOWS_CUDA_INDEX
    # Linux wheels on PyPI already bundle CUDA. macOS has no CUDA build.
    return None


def install(py: Path, args: argparse.Namespace) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    index = resolve_torch_index(args)
    torch_cmd = [str(py), "-m", "pip", "install", "torch"]
    if index:
        torch_cmd += ["--index-url", index]
        info(f"Installing torch from {index}")
    else:
        info("Installing torch from default PyPI index")
    try:
        run(torch_cmd)
    except BootstrapError:
        raise BootstrapError(
            "torch install failed. If this is a CUDA index mismatch, pick the "
            "build matching your driver at https://pytorch.org/get-started/locally/ "
            "and rerun with --torch-index <url>, or use --cpu for a CPU-only setup."
        ) from None

    run([str(py), "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements-dev.txt")])
    run([str(py), "-m", "pip", "install", "-e", str(REPO_ROOT)])


VERIFY_SNIPPET = """
import importlib, platform, sys
print("python        :", platform.python_version())
import torch
print("torch         :", torch.__version__)
cuda = torch.cuda.is_available()
print("cuda available:", cuda)
if cuda:
    print("cuda device   :", torch.cuda.get_device_name(0))
    print("cuda version  :", torch.version.cuda)
elif "+cpu" in torch.__version__:
    print("NOTE          : CPU-only torch wheel installed; GPU will not be used.")
import numpy
print("numpy         :", numpy.__version__)
for mod in ("rosa_compute", "exp0"):
    importlib.import_module(mod)
    print(f"import {mod:<7}: OK")
try:
    import rosa_soft
    print("rosa_soft     : OK")
except Exception as exc:
    print("rosa_soft     : FAILED -", exc)
    sys.exit(1)
"""


def verify(py: Path) -> None:
    info("Verifying environment")
    run([str(py), "-c", VERIFY_SNIPPET])


def _activation_files(venv_dir: Path) -> dict[Path, str]:
    """Map activation script path -> shell flavour."""
    files = {}
    if platform.system() == "Windows":
        files[venv_dir / "Scripts" / "activate.bat"] = "bat"
        files[venv_dir / "Scripts" / "Activate.ps1"] = "ps1"
    else:
        files[venv_dir / "bin" / "activate"] = "sh"
    return {p: kind for p, kind in files.items() if p.exists()}


def _render_block(pairs: list[tuple[str, str]], kind: str) -> str:
    lines = [BLOCK_START]
    for key, value in pairs:
        if kind == "bat":
            lines.append(f'set "{key}={value}"')
        elif kind == "ps1":
            lines.append(f'$env:{key} = "{value}"')
        else:
            lines.append(f'export {key}="{value}"')
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def persist_env(venv_dir: Path, pairs: list[tuple[str, str]]) -> None:
    if not pairs:
        return
    targets = _activation_files(venv_dir)
    if not targets:
        info("No activation scripts found; skipping --persist-env")
        return
    pattern = re.compile(
        re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n?",
        re.DOTALL,
    )
    for path, kind in targets.items():
        text = path.read_text(encoding="utf-8")
        text = pattern.sub("", text)  # idempotent: drop any previous block
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text + _render_block(pairs, kind), encoding="utf-8")
        info(f"Persisted {len(pairs)} variable(s) into {path.name}")


def parse_env_pairs(raw: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in raw:
        if "=" not in item:
            raise BootstrapError(f"--persist-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise BootstrapError(f"--persist-env got an empty key in {item!r}")
        pairs.append((key, value))
    return pairs


def print_next_steps(py: Path, pairs: list[tuple[str, str]]) -> None:
    if platform.system() == "Windows":
        activate = r".venv\Scripts\activate"
    else:
        activate = "source .venv/bin/activate"
    print()
    info("Done. Next steps:")
    print(f"    {activate}")
    print("    pytest -m \"exp0 and not slow\" -q")
    print("    python scripts/inspect_environment.py")
    print()
    print("  Optional environment variables:")
    print("    ROSA_MODEL_PATH      path to a ROSA checkpoint; enables the")
    print("                         'checkpoint'-marked tests (skipped otherwise)")
    print("    EXP0_RWKV7_CUDA_DIR  override the fused RWKV-7 CUDA source dir")
    if not pairs:
        print()
        print("  To persist one across sessions:")
        print("    python scripts/bootstrap_env.py --check "
              "--persist-env ROSA_MODEL_PATH=/path/to/model.pth")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the rosa-compute development environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install the CPU-only torch build instead of a CUDA build.",
    )
    parser.add_argument(
        "--torch-index",
        default=None,
        help="Explicit --index-url for torch (e.g. a different CUDA build). "
             "Overrides platform defaults and --cpu.",
    )
    parser.add_argument(
        "--no-venv",
        dest="use_venv",
        action="store_false",
        help="Install into the current interpreter instead of creating .venv.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing environment without installing anything.",
    )
    parser.add_argument(
        "--persist-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Write KEY=VALUE into the venv activation scripts so it is set on "
             "every activation. Repeatable. Idempotent.",
    )
    args = parser.parse_args()

    try:
        pairs = parse_env_pairs(args.persist_env)
        check_python()
        if args.check:
            py = venv_python(VENV_DIR) if VENV_DIR.exists() else Path(sys.executable)
            if not py.exists():
                raise BootstrapError(f"no environment found at {py}")
            verify(py)
        else:
            ensure_submodules()
            py = ensure_venv(args.use_venv)
            install(py, args)
            verify(py)
        persist_env(VENV_DIR, pairs)
        print_next_steps(py, pairs)
    except BootstrapError as exc:
        print(f"\n[bootstrap] ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[bootstrap] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
