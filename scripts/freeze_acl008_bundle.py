"""Regenerate ACL-008's analytic-only registry and final lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptive_correspondence.acl008 import (
    ACL008_LOCKED_FILES,
    build_analytic_registry,
    load_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    if tuple((repo / "evidence").glob("ACL-008-confirmatory-*.json")):
        raise RuntimeError("refusing to regenerate ACL-008 after evidence exists")
    bundle = repo / "preregistrations" / "ACL-008"
    manifest = load_manifest(bundle / "manifest.json")
    registry = build_analytic_registry(manifest)
    (bundle / "analytic_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lock = {
        "schema_version": 1,
        "experiment_id": "ACL-008",
        "kind": "preregistration-bundle-lock",
        "outcomes_generated": False,
        "scope": "Bundle files are byte-locked; the approved Git SHA freezes code/tests.",
        "files": {name: _sha256(bundle / name) for name in sorted(ACL008_LOCKED_FILES)},
    }
    (bundle / "LOCK.json").write_text(
        json.dumps(lock, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
