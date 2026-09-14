"""Synchronise the Jupytext source only when the notebook was edited more recently."""

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


def main() -> None:
    """Update analysis.py from its paired notebook when a manual notebook edit exists."""
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}

    root = project_root(Path(event.get("cwd", Path.cwd())))
    notebook_path = root / "analysis.ipynb"
    source_path = root / "analysis.py"
    if not notebook_path.exists() or not source_path.exists():
        return
    if notebook_path.stat().st_mtime_ns <= source_path.stat().st_mtime_ns:
        return

    uv_command = shutil.which("uv")
    if uv_command is None:
        raise RuntimeError("uv må være tilgjengelig for å synkronisere analysis.py")

    with tempfile.TemporaryDirectory(dir=root, prefix=".analysis-sync-") as temporary_directory:
        temporary_path = Path(temporary_directory) / "analysis.py"
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


if __name__ == "__main__":
    main()
