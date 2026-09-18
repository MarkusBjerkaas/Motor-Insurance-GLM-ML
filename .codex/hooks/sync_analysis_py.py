"""Synchronise paired Jupytext sources when their notebooks are newer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def project_root(cwd: Path) -> Path:
    """Resolve the Git root, regardless of the directory from which Codex starts."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def sync_pair(root: Path, notebook_name: str, source_name: str) -> None:
    """Update one percent-format source from its paired notebook if needed."""
    notebook_path = root / notebook_name
    source_path = root / source_name
    if not notebook_path.exists() or not source_path.exists():
        return
    if notebook_path.stat().st_mtime_ns <= source_path.stat().st_mtime_ns:
        return

    uv_command = shutil.which("uv")
    if uv_command is None:
        raise RuntimeError("uv må være tilgjengelig for å synkronisere Jupytext-kilder")

    with tempfile.TemporaryDirectory(
        dir=root, prefix=f".{source_path.stem}-sync-"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / source_name
        subprocess.run(
            [
                uv_command,
                "run",
                "jupytext",
                "--to",
                "py:percent",
                "--output",
                str(temporary_path),
                str(notebook_path),
            ],
            cwd=root,
            check=True,
        )
        temporary_path.replace(source_path)


def main() -> None:
    """Update all configured paired sources after a manual notebook edit."""
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}

    root = project_root(Path(event.get("cwd", Path.cwd())))
    for notebook_name, source_name in (
        ("analysis.ipynb", "analysis.py"),
        ("tweedie.ipynb", "tweedie.py"),
    ):
        sync_pair(root, notebook_name, source_name)


if __name__ == "__main__":
    main()
