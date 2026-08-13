"""Regenerate the analytic-only ACL-006 registry and lock, before outcomes only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptive_correspondence.acl006 import (
    ACL006_LOCKED_FILES,
    build_analytic_registry,
    load_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    evidence_matches = tuple((repo / "evidence").glob("ACL-006-confirmatory-*.json"))
    if evidence_matches:
        raise RuntimeError("refusing to regenerate ACL-006 bundle after evidence exists")
    bundle = repo / "preregistrations" / "ACL-006"
    manifest = load_manifest(bundle / "manifest.json")
    registry = build_analytic_registry(manifest)
    (bundle / "analytic_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    names = sorted(ACL006_LOCKED_FILES)
    lock = {
        "schema_version": 1,
        "experiment_id": "ACL-006",
        "kind": "preregistration-bundle-lock",
        "outcomes_generated": False,
        "scope": (
            "Bundle files are SHA-256 locked here; the approved Git commit freezes "
            "code and tests."
        ),
        "files": {name: _sha256(bundle / name) for name in names},
    }
    (bundle / "LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(bundle / "analytic_registry.json")
    print(bundle / "LOCK.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
