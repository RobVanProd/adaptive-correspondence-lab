"""Artifact serialization and reproducibility metadata."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .schema import json_safe


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"git_commit": None, "git_tracked_files_dirty": None}
    return {"git_commit": commit, "git_tracked_files_dirty": bool(status.strip())}


def provenance() -> dict[str, Any]:
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }
    metadata.update(_git_provenance())
    return metadata


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV artifact")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])
    temporary.replace(destination)
    return destination
