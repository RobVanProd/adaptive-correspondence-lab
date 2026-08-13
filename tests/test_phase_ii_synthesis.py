import hashlib
import json
from pathlib import Path


def test_phase_ii_outcome_hashes_are_live_and_terminal() -> None:
    payload = json.loads(Path("PHASE_II_OUTCOME.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "U3-correspondence-lattice"
    assert payload["termination_reached"] is True
    assert payload["universal_adaptive_process_supported"] is False
    for experiment in payload["experiments"]:
        artifact = Path(experiment["evidence_artifact"])
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == experiment[
            "evidence_sha256"
        ]
        report = Path(experiment["report_summary"])
        assert hashlib.sha256(report.read_bytes()).hexdigest() == experiment[
            "report_summary_sha256"
        ]


def test_synthesis_names_every_phase_ii_experiment() -> None:
    text = Path("PHASE_II_SYNTHESIS.md").read_text(encoding="utf-8")
    for identifier in ("ACL-006", "ACL-007", "ACL-008"):
        assert identifier in text
    assert "Outcome U3" in text
    assert "universal adaptive process" in text
