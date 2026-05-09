"""Guardrails and convenience helpers for demo notebook assets."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_NOTEBOOK_ROOT = REPO_ROOT / "docs" / "notebooks"
PROTECTED_SUFFIXES = {".ipynb", ".h5"}
DEFAULT_ALLOW_ENV = "PYAMPP_ALLOW_DEMO_NOTEBOOK_EDITS"


def _normalize_repo_path(path: str) -> str:
    return Path(path.replace("\\", "/")).as_posix().lstrip("./")


def is_protected_demo_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized.startswith("docs/notebooks/"):
        return False
    return Path(normalized).suffix.lower() in PROTECTED_SUFFIXES


def filter_protected_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if is_protected_demo_path(path)]


def git_changed_paths(repo_root: Path, *, staged: bool = False, against: str | None = None) -> list[str]:
    if staged and against:
        raise ValueError("staged and against are mutually exclusive")

    cmd = ["git", "-C", str(repo_root)]
    if staged:
        cmd.extend(["diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"])
    elif against:
        cmd.extend(["diff", "--name-only", "--diff-filter=ACMRTUXB", f"{against}...HEAD"])
    else:
        raise ValueError("Either staged=True or against=<revision> is required")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_changed_paths(paths: Sequence[str], *, allow_env: str = DEFAULT_ALLOW_ENV) -> list[str]:
    if os.getenv(allow_env):
        return []
    return filter_protected_paths(paths)


def copy_demo_bundle(source_root: Path, dest_root: Path) -> list[Path]:
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Demo notebook root does not exist: {source_root}")

    copied_files: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        rel = path.relative_to(source_root)
        target = dest_root / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IWUSR)
        copied_files.append(target)
    return copied_files


def find_demo_notebooks(root: Path) -> list[Path]:
    return sorted(path for path in Path(root).rglob("*.ipynb") if path.is_file())


def _open_notebook(path: Path) -> int:
    code = shutil.which("code")
    if code:
        return subprocess.run([code, str(path)], check=False).returncode

    if sys.platform == "darwin":
        result = subprocess.run(["open", "-a", "Visual Studio Code", str(path)], check=False)
        if result.returncode == 0:
            return 0

    print(f"Copied notebook is ready at: {path}")
    print("Open it manually or ensure the VS Code 'code' command is available.")
    return 0


def lock_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock tracked demo notebook assets to read-only in the current checkout.")
    parser.add_argument("--root", type=Path, default=DEMO_NOTEBOOK_ROOT, help="Demo notebook root.")
    args = parser.parse_args(argv)

    protected = [path for path in args.root.rglob("*") if path.is_file() and is_protected_demo_path(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))]
    for path in protected:
        mode = path.stat().st_mode
        path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    print(f"Locked {len(protected)} demo notebook asset(s) under {args.root}")
    return 0


def check_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block in-repo edits to demo notebook assets.")
    parser.add_argument("--staged", action="store_true", help="Check staged git changes (default).")
    parser.add_argument("--against", help="Check changes against a git revision, e.g. a merge-base SHA.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    parser.add_argument("--allow-env", default=DEFAULT_ALLOW_ENV, help="Environment variable override name.")
    args = parser.parse_args(argv)

    staged = args.staged or not args.against
    changed_paths = git_changed_paths(args.repo_root, staged=staged, against=args.against)
    blocked = check_changed_paths(changed_paths, allow_env=args.allow_env)

    if blocked:
        print("Demo notebook assets are read-only in-repo. Copy them outside the repo to edit:", file=sys.stderr)
        for path in blocked:
            print(f"  - {path}", file=sys.stderr)
        print("Use: pyampp-demo-notebooks play", file=sys.stderr)
        return 1
    return 0


def launch_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy the demo notebook bundle outside the repo and open it.")
    parser.add_argument("--source-root", type=Path, default=DEMO_NOTEBOOK_ROOT, help="Tracked notebook bundle root.")
    parser.add_argument("--dest-root", type=Path, help="Destination folder for the writable copy.")
    parser.add_argument("--notebook", help="Notebook file relative to the source root to open.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the copied notebook automatically.")
    args = parser.parse_args(argv)

    dest_root = args.dest_root or Path(tempfile.mkdtemp(prefix="pyampp-demo-notebook-"))
    dest_root.mkdir(parents=True, exist_ok=True)
    copy_demo_bundle(args.source_root, dest_root)

    notebooks = find_demo_notebooks(dest_root)
    if not notebooks:
        print(f"No notebook files were found under {args.source_root}.", file=sys.stderr)
        print(f"Bundle copied to: {dest_root}")
        return 1

    if args.notebook:
        notebook = (dest_root / args.notebook).resolve()
        if not notebook.exists():
            print(f"Requested notebook not found in the copied bundle: {args.notebook}", file=sys.stderr)
            return 1
    elif len(notebooks) == 1:
        notebook = notebooks[0]
    else:
        print("Multiple notebooks were found. Use --notebook to choose one:", file=sys.stderr)
        for path in notebooks:
            print(f"  - {path.relative_to(dest_root)}", file=sys.stderr)
        print(f"Bundle copied to: {dest_root}")
        return 1

    print(f"Bundle copied to: {dest_root}")
    print(f"Notebook to open: {notebook}")
    if not args.no_open:
        return _open_notebook(notebook)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        print("Usage: pyampp-demo-notebooks {check|lock|play} ...")
        return 1

    command, *rest = raw_argv
    if command == "check":
        return check_main(rest)
    if command == "lock":
        return lock_main(rest)
    if command == "play":
        return launch_main(rest)

    print(f"Unknown command: {command}", file=sys.stderr)
    print("Usage: pyampp-demo-notebooks {check|lock|play} ...", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())