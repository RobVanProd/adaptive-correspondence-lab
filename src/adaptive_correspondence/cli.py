"""Command-line entry point for reproducible, machine-readable experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .acl002 import execute_confirmatory, validate_preregistration_bundle
from .acl003 import (
    execute_confirmatory as execute_acl003_confirmatory,
)
from .acl003 import (
    validate_preregistration_bundle as validate_acl003_preregistration_bundle,
)
from .acl004 import (
    execute_confirmatory as execute_acl004_confirmatory,
)
from .acl004 import (
    validate_preregistration_bundle as validate_acl004_preregistration_bundle,
)
from .acl005 import (
    execute_confirmatory as execute_acl005_confirmatory,
)
from .acl005 import (
    validate_preregistration_bundle as validate_acl005_preregistration_bundle,
)
from .acl006 import (
    execute_confirmatory as execute_acl006_confirmatory,
)
from .acl006 import (
    validate_preregistration_bundle as validate_acl006_preregistration_bundle,
)
from .acl007 import (
    execute_confirmatory as execute_acl007_confirmatory,
)
from .acl007 import (
    validate_preregistration_bundle as validate_acl007_preregistration_bundle,
)
from .acl008 import (
    execute_confirmatory as execute_acl008_confirmatory,
)
from .acl008 import (
    validate_preregistration_bundle as validate_acl008_preregistration_bundle,
)
from .bandit import ContextualBandit, run_bandit_trajectory
from .experiments import (
    CategoricalExperimentConfig,
    run_equivalence,
    run_stability_sweep,
    run_transport_demo,
    software_verification,
)
from .gaussian import (
    DiagonalGaussianState,
    DiagonalQuadraticObjective,
    run_gaussian_trajectory,
)
from .io import provenance, write_csv, write_json


def _float_tuple(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _config(args: argparse.Namespace) -> CategoricalExperimentConfig:
    return CategoricalExperimentConfig(
        initial_state=args.initial,
        reward=args.reward,
        eta=args.eta,
        steps=args.steps,
        seed=args.seed,
    ).validated()


def _emit_json(payload: dict[str, Any], output: str | None) -> None:
    if output:
        destination = write_json(output, payload)
        print(destination.resolve())
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _write_sweep(payload: dict[str, Any], output: str | None) -> None:
    if output and Path(output).suffix.lower() == ".csv":
        destination = write_csv(output, payload["rows"])
        metadata_path = destination.with_suffix(".metadata.json")
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        write_json(metadata_path, metadata)
        print(destination.resolve())
        print(metadata_path.resolve())
        return
    _emit_json(payload, output)


def _add_categorical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial", type=_float_tuple, default=(0.2, 0.3, 0.5))
    parser.add_argument("--reward", type=_float_tuple, default=(0.7, -0.2, 0.1))
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1729)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acl",
        description="CPU-first adaptive-system correspondence experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    equivalence = subparsers.add_parser("equivalence", help="run exact categorical parity")
    _add_categorical_arguments(equivalence)
    equivalence.add_argument("--output")

    sweep = subparsers.add_parser("sweep", help="measure an epsilon-to-delta curve")
    _add_categorical_arguments(sweep)
    sweep.add_argument(
        "--perturbation",
        required=True,
        choices=[
            "euler",
            "reward-bias",
            "noise",
            "delay",
            "frequency",
            "mutation",
            "nonstationary",
            "finite-population",
            "constraint",
        ],
    )
    sweep.add_argument(
        "--epsilons",
        type=_float_tuple,
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
    )
    sweep.add_argument("--seeds", type=int, default=32)
    sweep.add_argument("--metric", choices=["endpoint-l1", "max-path-l1"], default="endpoint-l1")
    sweep.add_argument("--target-world", default="replicator-exact")
    sweep.add_argument("--output")

    transport = subparsers.add_parser(
        "transport", help="freeze a source slope and predict a target"
    )
    _add_categorical_arguments(transport)
    transport.add_argument("--perturbation", default="reward-bias")
    transport.add_argument("--source-epsilons", type=_float_tuple, default=(0.0005, 0.001, 0.002))
    transport.add_argument("--target-epsilons", type=_float_tuple, default=(0.003, 0.006, 0.012))
    transport.add_argument("--seeds", type=int, default=32)
    transport.add_argument("--output")

    gaussian = subparsers.add_parser("gaussian", help="run the pure Gaussian optimizer rung")
    gaussian.add_argument("--mean", type=_float_tuple, default=(1.5, -1.0))
    gaussian.add_argument("--std", type=_float_tuple, default=(0.8, 1.2))
    gaussian.add_argument("--target", type=_float_tuple, default=(0.0, 0.5))
    gaussian.add_argument("--curvature", type=_float_tuple, default=(1.0, 2.0))
    gaussian.add_argument("--eta", type=float, default=0.08)
    gaussian.add_argument("--steps", type=int, default=30)
    gaussian.add_argument("--mode", choices=["analytic", "rank-mu"], default="analytic")
    gaussian.add_argument("--samples", type=int, default=32)
    gaussian.add_argument("--parents", type=int)
    gaussian.add_argument("--seed", type=int, default=1729)
    gaussian.add_argument("--output")

    bandit = subparsers.add_parser("bandit", help="run the contextual-bandit NPG rung")
    bandit.add_argument("--eta", type=float, default=0.1)
    bandit.add_argument("--steps", type=int, default=30)
    bandit.add_argument("--mode", choices=["exact", "sampled"], default="exact")
    bandit.add_argument("--samples", type=int, default=128)
    bandit.add_argument("--seed", type=int, default=1729)
    bandit.add_argument("--output")

    verify = subparsers.add_parser("verify", help="run lightweight software checks")
    verify.add_argument("--output")

    demo = subparsers.add_parser("demo", help="write a compact suite of example artifacts")
    demo.add_argument("--output-dir", default="results/demo")
    demo.add_argument("--seed", type=int, default=1729)
    demo.add_argument("--seeds", type=int, default=32)

    acl002_validate = subparsers.add_parser(
        "acl002-validate",
        help="validate the locked ACL-002 bundle without generating outcomes",
    )
    acl002_validate.add_argument(
        "--bundle", default="preregistrations/ACL-002"
    )
    acl002_validate.add_argument("--output")

    acl002_run = subparsers.add_parser(
        "acl002-run",
        help="execute ACL-002 only after explicit approval of its preregistration SHA",
    )
    acl002_run.add_argument("--bundle", default="preregistrations/ACL-002")
    acl002_run.add_argument("--approved-sha", required=True)
    acl002_run.add_argument("--output", required=True)

    acl003_validate = subparsers.add_parser(
        "acl003-validate",
        help="validate the locked ACL-003 bundle without generating outcomes",
    )
    acl003_validate.add_argument("--bundle", default="preregistrations/ACL-003")
    acl003_validate.add_argument(
        "--reference-manifest", default="preregistrations/ACL-002/manifest.json"
    )
    acl003_validate.add_argument("--output")

    acl003_run = subparsers.add_parser(
        "acl003-run",
        help="execute ACL-003 only after explicit approval of its preregistration SHA",
    )
    acl003_run.add_argument("--bundle", default="preregistrations/ACL-003")
    acl003_run.add_argument(
        "--reference-manifest", default="preregistrations/ACL-002/manifest.json"
    )
    acl003_run.add_argument("--approved-sha", required=True)
    acl003_run.add_argument("--output", required=True)

    acl004_validate = subparsers.add_parser(
        "acl004-validate",
        help="validate the locked ACL-004 bundle without generating shadows",
    )
    acl004_validate.add_argument("--bundle", default="preregistrations/ACL-004")
    acl004_validate.add_argument("--output")

    acl004_run = subparsers.add_parser(
        "acl004-run",
        help="execute ACL-004 only after explicit approval of its preregistration SHA",
    )
    acl004_run.add_argument("--bundle", default="preregistrations/ACL-004")
    acl004_run.add_argument("--approved-sha", required=True)
    acl004_run.add_argument("--output", required=True)

    acl005_validate = subparsers.add_parser(
        "acl005-validate",
        help="validate the locked ACL-005 bundle without generating shadows",
    )
    acl005_validate.add_argument("--bundle", default="preregistrations/ACL-005")
    acl005_validate.add_argument("--output")

    acl005_run = subparsers.add_parser(
        "acl005-run",
        help="execute ACL-005 only after explicit approval of its preregistration SHA",
    )
    acl005_run.add_argument("--bundle", default="preregistrations/ACL-005")
    acl005_run.add_argument("--approved-sha", required=True)
    acl005_run.add_argument("--output", required=True)

    acl006_validate = subparsers.add_parser(
        "acl006-validate",
        help="validate the locked ACL-006 bundle without generating shadows",
    )
    acl006_validate.add_argument("--bundle", default="preregistrations/ACL-006")
    acl006_validate.add_argument("--output")

    acl006_run = subparsers.add_parser(
        "acl006-run",
        help="execute ACL-006 only after explicit approval of its preregistration SHA",
    )
    acl006_run.add_argument("--bundle", default="preregistrations/ACL-006")
    acl006_run.add_argument("--approved-sha", required=True)
    acl006_run.add_argument("--output", required=True)

    acl007_validate = subparsers.add_parser(
        "acl007-validate",
        help="validate the locked ACL-007 bundle without generating particles",
    )
    acl007_validate.add_argument("--bundle", default="preregistrations/ACL-007")
    acl007_validate.add_argument("--output")

    acl007_run = subparsers.add_parser(
        "acl007-run",
        help="execute ACL-007 only after explicit approval of its preregistration SHA",
    )
    acl007_run.add_argument("--bundle", default="preregistrations/ACL-007")
    acl007_run.add_argument("--approved-sha", required=True)
    acl007_run.add_argument("--output", required=True)

    acl008_validate = subparsers.add_parser(
        "acl008-validate",
        help="validate the locked ACL-008 bundle without perturbed trajectories",
    )
    acl008_validate.add_argument("--bundle", default="preregistrations/ACL-008")
    acl008_validate.add_argument(
        "--reference-manifest", default="preregistrations/ACL-003/manifest.json"
    )
    acl008_validate.add_argument("--output")

    acl008_run = subparsers.add_parser(
        "acl008-run",
        help="execute ACL-008 only after explicit approval of its preregistration SHA",
    )
    acl008_run.add_argument("--bundle", default="preregistrations/ACL-008")
    acl008_run.add_argument(
        "--reference-manifest", default="preregistrations/ACL-003/manifest.json"
    )
    acl008_run.add_argument("--approved-sha", required=True)
    acl008_run.add_argument("--output", required=True)
    return parser


def _gaussian_payload(args: argparse.Namespace) -> dict[str, Any]:
    standard_deviation = np.asarray(args.std, dtype=np.float64)
    if np.any(standard_deviation <= 0.0):
        raise ValueError("--std values must be strictly positive")
    trajectory = run_gaussian_trajectory(
        initial_state=DiagonalGaussianState(args.mean, np.log(standard_deviation)),
        objective=DiagonalQuadraticObjective(args.target, args.curvature),
        eta=args.eta,
        steps=args.steps,
        mode=args.mode,
        seed=args.seed,
        sample_count=args.samples,
        parent_count=args.parents,
    )
    return {
        "kind": "gaussian-natural-gradient-run",
        "scientific_status": "reference trajectory, not a performance claim",
        "trajectory": trajectory.to_dict(),
        "provenance": provenance(),
    }


def _bandit_payload(args: argparse.Namespace) -> dict[str, Any]:
    bandit = ContextualBandit(
        rewards=((0.8, 0.1, -0.2), (-0.1, 0.4, 0.9)),
        context_probabilities=(0.6, 0.4),
    )
    initial_logits = np.zeros((2, 3), dtype=np.float64)
    trajectory = run_bandit_trajectory(
        bandit=bandit,
        initial_logits=initial_logits,
        eta=args.eta,
        steps=args.steps,
        mode=args.mode,
        seed=args.seed,
        sample_count=args.samples,
    )
    return {
        "kind": "contextual-bandit-natural-policy-gradient-run",
        "scientific_status": "reference trajectory, not an RL benchmark",
        "trajectory": trajectory.to_dict(),
        "provenance": provenance(),
    }


def _demo(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CategoricalExperimentConfig(seed=args.seed)
    created: list[str] = []

    equivalence = write_json(output_dir / "equivalence.json", run_equivalence(config))
    created.append(str(equivalence.resolve()))
    for kind in ("mutation", "euler"):
        epsilons = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
        payload = run_stability_sweep(
            config,
            perturbation_kind=kind,
            epsilons=epsilons,
            seed_count=args.seeds,
        )
        csv_path = write_csv(output_dir / f"{kind}.csv", payload["rows"])
        metadata_path = csv_path.with_suffix(".metadata.json")
        write_json(metadata_path, {key: value for key, value in payload.items() if key != "rows"})
        created.extend((str(csv_path.resolve()), str(metadata_path.resolve())))
    transport = write_json(
        output_dir / "transport.json",
        run_transport_demo(config, seed_count=args.seeds),
    )
    created.append(str(transport.resolve()))

    gaussian_args = argparse.Namespace(
        mean=(1.5, -1.0),
        std=(0.8, 1.2),
        target=(0.0, 0.5),
        curvature=(1.0, 2.0),
        eta=0.08,
        steps=20,
        mode="analytic",
        samples=32,
        parents=None,
        seed=args.seed,
    )
    gaussian = write_json(output_dir / "gaussian.json", _gaussian_payload(gaussian_args))
    created.append(str(gaussian.resolve()))

    bandit_args = argparse.Namespace(
        eta=0.1,
        steps=20,
        mode="exact",
        samples=128,
        seed=args.seed,
    )
    bandit = write_json(output_dir / "bandit.json", _bandit_payload(bandit_args))
    created.append(str(bandit.resolve()))
    verification = write_json(output_dir / "verification.json", software_verification())
    created.append(str(verification.resolve()))
    return {"kind": "demo-suite", "created": created, "provenance": provenance()}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "equivalence":
            _emit_json(run_equivalence(_config(args)), args.output)
        elif args.command == "sweep":
            payload = run_stability_sweep(
                _config(args),
                perturbation_kind=args.perturbation,
                epsilons=args.epsilons,
                seed_count=args.seeds,
                metric=args.metric,
                target_world=args.target_world,
            )
            _write_sweep(payload, args.output)
        elif args.command == "transport":
            payload = run_transport_demo(
                _config(args),
                perturbation_kind=args.perturbation,
                source_epsilons=args.source_epsilons,
                target_epsilons=args.target_epsilons,
                seed_count=args.seeds,
            )
            _emit_json(payload, args.output)
        elif args.command == "gaussian":
            _emit_json(_gaussian_payload(args), args.output)
        elif args.command == "bandit":
            _emit_json(_bandit_payload(args), args.output)
        elif args.command == "verify":
            payload = software_verification()
            _emit_json(payload, args.output)
            return 0 if payload["passed"] else 1
        elif args.command == "demo":
            print(json.dumps(_demo(args), indent=2, sort_keys=True))
        elif args.command == "acl002-validate":
            payload = validate_preregistration_bundle(args.bundle)
            _emit_json(payload, args.output)
        elif args.command == "acl002-run":
            bundle = Path(args.bundle)
            destination = execute_confirmatory(
                repo_path=Path.cwd(),
                manifest_path=bundle / "manifest.json",
                registry_path=bundle / "analytic_registry.json",
                lock_path=bundle / "LOCK.json",
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl003-validate":
            payload = validate_acl003_preregistration_bundle(
                args.bundle, reference_path=args.reference_manifest
            )
            _emit_json(payload, args.output)
        elif args.command == "acl003-run":
            destination = execute_acl003_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                reference_path=args.reference_manifest,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl004-validate":
            payload = validate_acl004_preregistration_bundle(args.bundle)
            _emit_json(payload, args.output)
        elif args.command == "acl004-run":
            destination = execute_acl004_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl005-validate":
            payload = validate_acl005_preregistration_bundle(args.bundle)
            _emit_json(payload, args.output)
        elif args.command == "acl005-run":
            destination = execute_acl005_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl006-validate":
            payload = validate_acl006_preregistration_bundle(args.bundle)
            _emit_json(payload, args.output)
        elif args.command == "acl006-run":
            destination = execute_acl006_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl007-validate":
            payload = validate_acl007_preregistration_bundle(args.bundle)
            _emit_json(payload, args.output)
        elif args.command == "acl007-run":
            destination = execute_acl007_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        elif args.command == "acl008-validate":
            payload = validate_acl008_preregistration_bundle(
                args.bundle, reference_path=args.reference_manifest
            )
            _emit_json(payload, args.output)
        elif args.command == "acl008-run":
            destination = execute_acl008_confirmatory(
                repo_path=Path.cwd(),
                bundle_path=args.bundle,
                reference_path=args.reference_manifest,
                approved_sha=args.approved_sha,
                output_path=args.output,
            )
            print(destination.resolve())
        else:
            parser.error(f"unhandled command: {args.command}")
    except (ValueError, FloatingPointError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
