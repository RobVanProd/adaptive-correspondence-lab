import csv
import json
from pathlib import Path

from adaptive_correspondence.cli import main


def test_verify_command_writes_passing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "verify.json"
    assert main(["verify", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_sweep_csv_has_companion_metadata(tmp_path: Path) -> None:
    output = tmp_path / "curve.csv"
    return_code = main(
        [
            "sweep",
            "--perturbation",
            "mutation",
            "--epsilons",
            "0,0.01",
            "--seeds",
            "2",
            "--steps",
            "3",
            "--output",
            str(output),
        ]
    )
    assert return_code == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["perturbation"] == "mutation"


def test_invalid_gaussian_std_returns_usage_error(capsys) -> None:
    return_code = main(["gaussian", "--std=-1,1"])
    assert return_code == 2
    assert "strictly positive" in capsys.readouterr().err
