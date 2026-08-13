import csv
import json
from pathlib import Path

import adaptive_correspondence.cli as cli_module
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


def test_acl003_validation_command_never_invokes_runner(monkeypatch, capsys) -> None:
    expected = {"valid": True, "outcomes_generated": False}
    monkeypatch.setattr(
        cli_module,
        "validate_acl003_preregistration_bundle",
        lambda bundle, reference_path: {
            **expected,
            "bundle": bundle,
            "reference": reference_path,
        },
    )

    return_code = main(
        [
            "acl003-validate",
            "--bundle",
            "frozen",
            "--reference-manifest",
            "reference.json",
        ]
    )

    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcomes_generated"] is False
    assert payload["bundle"] == "frozen"


def test_acl004_validation_command_never_invokes_runner(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "validate_acl004_preregistration_bundle",
        lambda bundle: {"valid": True, "outcomes_generated": False, "bundle": bundle},
    )
    return_code = main(["acl004-validate", "--bundle", "gaussian-frozen"])
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "bundle": "gaussian-frozen",
        "outcomes_generated": False,
        "valid": True,
    }


def test_acl006_validation_command_never_invokes_runner(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "validate_acl006_preregistration_bundle",
        lambda bundle: {"valid": True, "outcomes_generated": False, "bundle": bundle},
    )
    return_code = main(["acl006-validate", "--bundle", "support-frozen"])
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "bundle": "support-frozen",
        "outcomes_generated": False,
        "valid": True,
    }


def test_acl007_validation_command_never_invokes_runner(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "validate_acl007_preregistration_bundle",
        lambda bundle: {"valid": True, "outcomes_generated": False, "bundle": bundle},
    )
    return_code = main(["acl007-validate", "--bundle", "inference-frozen"])
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "bundle": "inference-frozen",
        "outcomes_generated": False,
        "valid": True,
    }


def test_acl008_validation_command_never_invokes_runner(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "validate_acl008_preregistration_bundle",
        lambda bundle, reference_path: {
            "valid": True,
            "outcomes_generated": False,
            "bundle": bundle,
            "reference": reference_path,
        },
    )
    return_code = main(
        [
            "acl008-validate",
            "--bundle",
            "burg-frozen",
            "--reference-manifest",
            "source.json",
        ]
    )
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcomes_generated"] is False
    assert payload["bundle"] == "burg-frozen"
