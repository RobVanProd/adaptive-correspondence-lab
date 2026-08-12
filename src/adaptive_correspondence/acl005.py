"""ACL-005 cross-class transport protocol for contextual-bandit NPG shadows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acl002 import (
    assert_execution_context,
    git_execution_state,
    sha256_file,
    type7_quantile,
    validate_lock,
)
from .control_finite_sample_bridge import (
    ControlBridgeState,
    exact_natural_direction,
    joint_fisher_cosine,
    sample_plugin_npg_shadows,
)
from .io import provenance, write_json

FloatArray = NDArray[np.float64]

ACL005_CONTEXTS = 2
ACL005_ACTIONS = 3
ACL005_INTERACTION_SAMPLE_COUNT = 128
ACL005_FISHER_RCOND = 1e-12
ACL005_REPLICATION_SCHEDULE = (4096, 8192, 16384, 32768, 65536)
ACL005_CHUNK_SIZE = 2048
ACL005_HALF_COSINE_MIN = 0.98
ACL005_H2_COSINE_MIN = 0.99
ACL005_H1_SHADOW_COUNT = 2048
ACL005_REGULAR_MIN_EXPECTED_CELL_COUNT = 4.0
ACL005_STRESS_MAX_EXPECTED_CELL_COUNT = 0.75
ACL005_SOURCE_EXPERIMENT = "ACL-004"
ACL005_SOURCE_RULE = "unchanged-block-fisher-cosine-law"
ACL005_SOURCE_EVIDENCE = {
    "approved_preregistration_sha": "3ba4be7ce1460a40c4ef0879018df58947c36edb",
    "evidence_commit": "355dd97472da4230eff877b9a3c8c7c4626057cd",
    "evidence_artifact": (
        "evidence/ACL-004-confirmatory-"
        "3ba4be7ce1460a40c4ef0879018df58947c36edb.json"
    ),
    "evidence_sha256": "3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a",
    "report_commit": "5a3ef122a04bd6ec59aaa77c2493a5a9c0979f7a",
    "report_summary": "analysis/ACL-004-confirmatory/summary.json",
    "report_summary_sha256": (
        "b4d9864b6ab935aa39bc98ab1c144e13030ebad57931eaf4bc1cbcbaf3d2e019"
    ),
    "source_h2_verdict": "PASS",
    "source_all_landscapes_converged": True,
    "source_minimum_mean_block_fisher_cosine": 0.9999523237518517,
    "source_minimum_covariance_block_fisher_cosine": 0.9995521999195026,
}
ACL005_LOCKED_FILES = frozenset(
    {
        "ANALYSIS_PLAN.md",
        "DERIVATION.md",
        "PREREGISTRATION.md",
        "README.md",
        "analytic_registry.json",
        "manifest.json",
    }
)


@dataclass(frozen=True)
class ACL005Landscape:
    identifier: str
    role: str
    state: ControlBridgeState
    seed: int


@dataclass(frozen=True)
class ACL005Manifest:
    experiment_id: str
    contexts: int
    actions: int
    interaction_sample_count: int
    empirical_fisher_rcond: float
    replication_schedule: tuple[int, ...]
    chunk_size: int
    half_cosine_min: float
    h2_cosine_min: float
    h1_shadow_count: int
    regular_min_expected_cell_count: float
    stress_max_expected_cell_count: float
    source_experiment: str
    source_rule: str
    landscapes: tuple[ACL005Landscape, ...]
    benchmark_scope: str
    inference_scope: str
    transport_scope: str
    raw: dict[str, Any]


def _integer(payload: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _positive_float(payload: dict[str, Any], key: str) -> float:
    value = float(payload.get(key))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _minimum_expected_cell_count(
    state: ControlBridgeState, sample_count: int
) -> float:
    _, contexts, _, policy = state.arrays()
    return float(sample_count * np.min(contexts[:, None] * policy))


def validate_manifest_dict(payload: dict[str, Any]) -> ACL005Manifest:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ACL-005 manifest schema")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if payload.get("randomness") != "PCG64-independent-landscape-streams":
        raise ValueError("ACL-005 randomness mismatch")
    contexts = _integer(payload, "contexts", minimum=2)
    actions = _integer(payload, "actions", minimum=2)
    sample_count = _integer(payload, "interaction_sample_count")
    rcond = _positive_float(payload, "empirical_fisher_rcond")
    schedule_payload = payload.get("replication_schedule")
    if not isinstance(schedule_payload, list) or not schedule_payload:
        raise ValueError("replication_schedule must be non-empty")
    schedule = tuple(schedule_payload)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in schedule):
        raise ValueError("replication schedule must contain integers")
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError("replication schedule must be strictly increasing")
    chunk_size = _integer(payload, "shadow_chunk_size")
    if any(value % (2 * chunk_size) != 0 for value in schedule):
        raise ValueError("each checkpoint and half must align to complete chunks")
    half_cosine_min = _positive_float(payload, "half_convergence_fisher_cosine_min")
    h2_cosine_min = _positive_float(payload, "h2_fisher_cosine_min")
    if half_cosine_min > 1.0 or h2_cosine_min > 1.0:
        raise ValueError("cosine thresholds cannot exceed one")
    h1_shadow_count = _integer(payload, "h1_shadow_count")
    if h1_shadow_count > schedule[0] or h1_shadow_count % chunk_size != 0:
        raise ValueError("H1 shadow count must align within the first checkpoint")
    regular_minimum = _positive_float(payload, "regular_min_expected_cell_count")
    stress_maximum = _positive_float(payload, "stress_max_expected_cell_count")
    source_experiment = payload.get("source_experiment")
    source_rule = payload.get("source_rule")
    if not isinstance(source_experiment, str) or not source_experiment:
        raise ValueError("source_experiment must be non-empty")
    if not isinstance(source_rule, str) or not source_rule:
        raise ValueError("source_rule must be non-empty")

    items = payload.get("landscapes")
    if not isinstance(items, list) or not items:
        raise ValueError("landscapes must be a non-empty list")
    identifiers: set[str] = set()
    seeds: set[int] = set()
    landscapes = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each ACL-005 landscape must be an object")
        identifier = item.get("id")
        role = item.get("role")
        seed = item.get("seed")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("landscape IDs must be non-empty and unique")
        if role not in {"confirmatory-target", "stress-target"}:
            raise ValueError("landscape role must be confirmatory-target or stress-target")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in seeds:
            raise ValueError("landscape seeds must be distinct non-negative integers")
        state = ControlBridgeState(
            rewards=item.get("rewards"),
            context_probabilities=item.get("context_probabilities"),
            logits=item.get("logits"),
        )
        rewards, context_probabilities, _, _ = state.arrays()
        if rewards.shape != (contexts, actions):
            raise ValueError("landscape dimensions differ from manifest")
        if context_probabilities.size != contexts:
            raise ValueError("context count differs from manifest")
        expected_minimum = _minimum_expected_cell_count(state, sample_count)
        if role == "confirmatory-target" and expected_minimum < regular_minimum:
            raise ValueError("regular landscape violates minimum expected cell count")
        if role == "stress-target" and expected_minimum > stress_maximum:
            raise ValueError("stress landscape violates maximum expected cell count")
        identifiers.add(identifier)
        seeds.add(seed)
        landscapes.append(ACL005Landscape(identifier, role, state, seed))

    scopes = tuple(
        payload.get(key) for key in ("benchmark_scope", "inference_scope", "transport_scope")
    )
    if not all(isinstance(value, str) and value for value in scopes):
        raise ValueError("ACL-005 scope fields must be non-empty")

    if experiment_id == "ACL-005":
        regular_ids = tuple(
            landscape.identifier
            for landscape in landscapes
            if landscape.role == "confirmatory-target"
        )
        stress_ids = tuple(
            landscape.identifier for landscape in landscapes if landscape.role == "stress-target"
        )
        exact = (
            contexts == ACL005_CONTEXTS
            and actions == ACL005_ACTIONS
            and sample_count == ACL005_INTERACTION_SAMPLE_COUNT
            and rcond == ACL005_FISHER_RCOND
            and schedule == ACL005_REPLICATION_SCHEDULE
            and chunk_size == ACL005_CHUNK_SIZE
            and half_cosine_min == ACL005_HALF_COSINE_MIN
            and h2_cosine_min == ACL005_H2_COSINE_MIN
            and h1_shadow_count == ACL005_H1_SHADOW_COUNT
            and regular_minimum == ACL005_REGULAR_MIN_EXPECTED_CELL_COUNT
            and stress_maximum == ACL005_STRESS_MAX_EXPECTED_CELL_COUNT
            and source_experiment == ACL005_SOURCE_EXPERIMENT
            and source_rule == ACL005_SOURCE_RULE
            and payload.get("source_evidence") == ACL005_SOURCE_EVIDENCE
            and regular_ids == tuple(f"R{index:02d}" for index in range(1, 11))
            and stress_ids == tuple(f"S{index:02d}" for index in range(1, 5))
            and scopes
            == (
                "deterministic-held-out-contextual-bandit-benchmark",
                "descriptive-criteria-not-population-confidence",
                "gaussian-to-control-unchanged-normalized-direction-law",
            )
        )
        if not exact:
            raise ValueError("ACL-005 design constants mismatch")
    return ACL005Manifest(
        experiment_id=experiment_id,
        contexts=contexts,
        actions=actions,
        interaction_sample_count=sample_count,
        empirical_fisher_rcond=rcond,
        replication_schedule=schedule,
        chunk_size=chunk_size,
        half_cosine_min=half_cosine_min,
        h2_cosine_min=h2_cosine_min,
        h1_shadow_count=h1_shadow_count,
        regular_min_expected_cell_count=regular_minimum,
        stress_max_expected_cell_count=stress_maximum,
        source_experiment=source_experiment,
        source_rule=source_rule,
        landscapes=tuple(landscapes),
        benchmark_scope=scopes[0],
        inference_scope=scopes[1],
        transport_scope=scopes[2],
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL005Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-005 manifest") from error
    return validate_manifest_dict(payload)


def _context_fisher_norms(state: ControlBridgeState, direction: FloatArray) -> list[float]:
    _, contexts, _, policy = state.arrays()
    values = []
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        values.append(float(np.sqrt(direction[context] @ fisher @ direction[context])))
    return values


def build_analytic_registry(manifest: ACL005Manifest) -> dict[str, Any]:
    entries = []
    for landscape in manifest.landscapes:
        direction = exact_natural_direction(landscape.state)
        norms = _context_fisher_norms(landscape.state, direction)
        if min(norms) <= 1e-12:
            raise ValueError("ACL-005 analytic context block is too small")
        minimum = _minimum_expected_cell_count(
            landscape.state, manifest.interaction_sample_count
        )
        entries.append(
            {
                "id": landscape.identifier,
                "role": landscape.role,
                "seed": landscape.seed,
                "stratum": (
                    "regular" if landscape.role == "confirmatory-target" else "stress"
                ),
                "analytic_direction": direction.tolist(),
                "context_fisher_norms": norms,
                "minimum_expected_joint_cell_count_per_shadow": minimum,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "analytic-contextual-bandit-npg-registry",
        "outcomes_generated": False,
        "shadow_count": 0,
        "comparator_source": "exact-policy-gradient-and-categorical-fisher",
        "landscapes": entries,
    }


def reproduce_stopped_mean(result: dict[str, Any]) -> FloatArray:
    chunks = result.get("shadow_chunks", [])
    if not chunks:
        raise ValueError("landscape result has no shadow chunks")
    count = sum(int(chunk["count"]) for chunk in chunks)
    total = np.sum(
        np.asarray([chunk["direction_sum"] for chunk in chunks], dtype=np.float64), axis=0
    )
    return total / count


def _mean_from_chunks(chunks: list[dict[str, Any]]) -> FloatArray:
    return reproduce_stopped_mean({"shadow_chunks": chunks})


def _uncertainty_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(chunk["count"]) for chunk in chunks)
    if count < 2:
        raise ValueError("shadow uncertainty requires at least two observations")
    mean = _mean_from_chunks(chunks)
    flat_mean = mean.ravel()
    outer = np.sum(
        np.asarray([chunk["direction_outer_sum"] for chunk in chunks], dtype=np.float64),
        axis=0,
    )
    covariance = (outer - count * np.outer(flat_mean, flat_mean)) / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-14) or not np.all(np.isfinite(covariance)):
        raise FloatingPointError("invalid shadow covariance from sufficient statistics")
    return {
        "sample_covariance": covariance.tolist(),
        "coordinate_standard_error_of_mean": np.sqrt(
            np.maximum(diagonal, 0.0) / count
        ).reshape(mean.shape).tolist(),
    }


def _many_context_cosines(
    state: ControlBridgeState, directions: FloatArray, analytic: FloatArray
) -> list[list[float | None]]:
    _, contexts, _, policy = state.arrays()
    output: list[list[float | None]] = []
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        values = directions[:, context, :]
        target = analytic[context]
        numerators = np.einsum("bi,ij,j->b", values, fisher, target)
        left_norms = np.einsum("bi,ij,bj->b", values, fisher, values)
        target_norm = float(target @ fisher @ target)
        denominators = np.sqrt(np.maximum(left_norms, 0.0) * target_norm)
        cosines: list[float | None] = []
        for numerator, denominator in zip(numerators, denominators, strict=True):
            if denominator <= 0.0:
                cosines.append(None)
            else:
                cosines.append(float(np.clip(numerator / denominator, -1.0, 1.0)))
        output.append(cosines)
    return output


def _safe_context_cosines(
    state: ControlBridgeState, left: FloatArray, right: FloatArray
) -> list[float | None]:
    rewards, contexts, _, policy = state.arrays()
    left_array = np.asarray(left, dtype=np.float64).reshape(rewards.shape)
    right_array = np.asarray(right, dtype=np.float64).reshape(rewards.shape)
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("control directions must be finite")
    output: list[float | None] = []
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        numerator = float(left_array[context] @ fisher @ right_array[context])
        left_norm = float(left_array[context] @ fisher @ left_array[context])
        right_norm = float(right_array[context] @ fisher @ right_array[context])
        denominator = float(np.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0)))
        output.append(
            None
            if denominator <= 0.0
            else float(np.clip(numerator / denominator, -1.0, 1.0))
        )
    return output


def _safe_joint_cosine(
    state: ControlBridgeState, left: FloatArray, right: FloatArray
) -> float | None:
    try:
        return joint_fisher_cosine(state, left, right)
    except ValueError as error:
        if "nonzero directions" not in str(error):
            raise
        return None


def estimate_landscape(
    manifest: ACL005Manifest, landscape: ACL005Landscape
) -> dict[str, Any]:
    analytic = exact_natural_direction(landscape.state)
    rng = np.random.Generator(np.random.PCG64(landscape.seed))
    chunks: list[dict[str, Any]] = []
    h1_context_cosines: list[list[float | None]] = [
        [] for _ in range(manifest.contexts)
    ]
    history = []
    converged = False
    stopped = manifest.replication_schedule[-1]
    checkpoints = set(manifest.replication_schedule)
    generated = 0
    while generated < manifest.replication_schedule[-1]:
        shadows = sample_plugin_npg_shadows(
            landscape.state,
            sample_count=manifest.interaction_sample_count,
            replications=manifest.chunk_size,
            rng=rng,
            batch_size=manifest.chunk_size,
            rcond=manifest.empirical_fisher_rcond,
        )
        flat = shadows.reshape(manifest.chunk_size, -1)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "count": manifest.chunk_size,
                "direction_sum": np.sum(shadows, axis=0).tolist(),
                "direction_outer_sum": (flat.T @ flat).tolist(),
            }
        )
        if generated < manifest.h1_shadow_count:
            remaining = manifest.h1_shadow_count - generated
            values = _many_context_cosines(landscape.state, shadows[:remaining], analytic)
            for context, context_values in enumerate(values):
                h1_context_cosines[context].extend(context_values)
        generated += manifest.chunk_size
        if generated not in checkpoints:
            continue
        half_chunk_count = len(chunks) // 2
        first_mean = _mean_from_chunks(chunks[:half_chunk_count])
        second_mean = _mean_from_chunks(chunks[half_chunk_count:])
        half_cosines = _safe_context_cosines(landscape.state, first_mean, second_mean)
        history.append(
            {
                "replications": generated,
                "first_half_mean": first_mean.tolist(),
                "second_half_mean": second_mean.tolist(),
                "half_context_fisher_cosines": half_cosines,
            }
        )
        if all(
            value is not None and value >= manifest.half_cosine_min
            for value in half_cosines
        ):
            converged = True
            stopped = generated
            break
    final_mean = _mean_from_chunks(chunks)
    final_context = _safe_context_cosines(landscape.state, final_mean, analytic)
    return {
        "landscape_id": landscape.identifier,
        "role": landscape.role,
        "seed": landscape.seed,
        "converged": converged,
        "stopped_replications": stopped,
        "analytic_direction": analytic.tolist(),
        "stopped_mean_direction": final_mean.tolist(),
        "final_context_cosines": final_context,
        "joint_cosine": _safe_joint_cosine(landscape.state, final_mean, analytic),
        "checkpoint_history": history,
        "shadow_chunks": chunks,
        "shadow_uncertainty": _uncertainty_from_chunks(chunks),
        "rng_state_after": rng.bit_generator.state,
        "h1_context_cosines": h1_context_cosines,
    }


def _h1_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for context, raw_values in enumerate(result["h1_context_cosines"]):
        values = [float(value) for value in raw_values if value is not None]
        undefined = len(raw_values) - len(values)
        row: dict[str, Any] = {
            "landscape_id": result["landscape_id"],
            "role": result["role"],
            "context": context,
            "count": len(raw_values),
            "defined_count": len(values),
            "undefined_count": undefined,
        }
        if values:
            row.update(
                {
                    "q10": type7_quantile(values, 0.1),
                    "median": type7_quantile(values, 0.5),
                    "q90": type7_quantile(values, 0.9),
                    "fraction_positive_among_defined": sum(value > 0.0 for value in values)
                    / len(values),
                }
            )
        else:
            row.update(
                {
                    "q10": None,
                    "median": None,
                    "q90": None,
                    "fraction_positive_among_defined": None,
                }
            )
        rows.append(row)
    return rows


def analyze_landscape_results(
    manifest: ACL005Manifest, results: list[dict[str, Any]]
) -> dict[str, Any]:
    if {row["landscape_id"] for row in results} != {
        landscape.identifier for landscape in manifest.landscapes
    }:
        raise ValueError("ACL-005 result landscape IDs differ from manifest")
    expected_roles = {
        landscape.identifier: landscape.role for landscape in manifest.landscapes
    }
    if any(expected_roles[row["landscape_id"]] != row["role"] for row in results):
        raise ValueError("ACL-005 result roles differ from manifest")
    regular = [row for row in results if row["role"] == "confirmatory-target"]
    if not regular:
        raise ValueError("ACL-005 requires at least one regular target")
    all_regular_converged = all(row["converged"] for row in regular)
    regular_cosines = [
        value for row in regular for value in row["final_context_cosines"]
    ]
    minimum = (
        None
        if any(value is None for value in regular_cosines)
        else min(float(value) for value in regular_cosines if value is not None)
    )
    if not all_regular_converged:
        verdict = "INCONCLUSIVE"
    elif minimum is not None and minimum >= manifest.h2_cosine_min:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "experiment_id": manifest.experiment_id,
        "transport_verdict": verdict,
        "all_regular_landscapes_converged": all_regular_converged,
        "convergence_threshold": manifest.half_cosine_min,
        "transport_threshold": manifest.h2_cosine_min,
        "regular_minimum_context_fisher_cosine": minimum,
        "stress_gating": False,
        "joint_cosine_gating": False,
        "target_refit": False,
        "landscape_results": [
            {
                "landscape_id": row["landscape_id"],
                "role": row["role"],
                "converged": row["converged"],
                "stopped_replications": row["stopped_replications"],
                "final_context_cosines": row["final_context_cosines"],
                "joint_cosine": row["joint_cosine"],
            }
            for row in results
        ],
        "h1_descriptive": [entry for row in results for entry in _h1_summary(row)],
    }


def validate_preregistration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if (
        lock.get("experiment_id") != "ACL-005"
        or lock.get("kind") != "preregistration-bundle-lock"
        or lock.get("outcomes_generated") is not False
        or set(lock.get("files", {})) != ACL005_LOCKED_FILES
    ):
        raise ValueError("ACL-005 lock must contain the exact frozen file set")
    actual_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if actual_files != ACL005_LOCKED_FILES | {"LOCK.json"}:
        raise ValueError("ACL-005 bundle must have exact frozen directory contents")
    manifest = load_manifest(bundle / "manifest.json")
    try:
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-005 analytic registry") from error
    if locked_registry != build_analytic_registry(manifest):
        raise ValueError("ACL-005 analytic registry does not match clean recomputation")
    regular = sum(
        landscape.role == "confirmatory-target" for landscape in manifest.landscapes
    )
    return {
        "schema_version": 1,
        "experiment_id": "ACL-005",
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "regular_landscape_count": regular,
        "stress_landscape_count": len(manifest.landscapes) - regular,
        "interaction_sample_count": manifest.interaction_sample_count,
        "replication_schedule": list(manifest.replication_schedule),
        "locked_file_count": len(lock["files"]),
    }


def validate_source_evidence(
    repo_path: str | Path, manifest: ACL005Manifest
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    source = manifest.raw.get("source_evidence")
    if source != ACL005_SOURCE_EVIDENCE:
        raise ValueError("ACL-005 source evidence differs from the frozen source record")
    evidence = (repo / source["evidence_artifact"]).resolve()
    report = (repo / source["report_summary"]).resolve()
    try:
        evidence.relative_to(repo)
        report.relative_to(repo)
    except ValueError as error:
        raise ValueError("ACL-005 source evidence paths must stay inside the repository") from error
    try:
        evidence_hash = sha256_file(evidence)
        report_hash = sha256_file(report)
    except OSError as error:
        raise ValueError("cannot read frozen ACL-004 source evidence") from error
    if evidence_hash != source["evidence_sha256"]:
        raise ValueError("ACL-004 source evidence SHA-256 mismatch")
    if report_hash != source["report_summary_sha256"]:
        raise ValueError("ACL-004 source report SHA-256 mismatch")
    return {
        "valid": True,
        "source_experiment": manifest.source_experiment,
        "evidence_artifact": source["evidence_artifact"],
        "evidence_sha256": source["evidence_sha256"],
        "report_summary": source["report_summary"],
        "report_summary_sha256": source["report_summary_sha256"],
    }


def execute_confirmatory(
    *,
    repo_path: str | Path,
    bundle_path: str | Path,
    approved_sha: str,
    output_path: str | Path,
) -> Path:
    repo = Path(repo_path).resolve()
    requested = Path(output_path)
    if not requested.is_absolute():
        requested = repo / requested
    requested = requested.resolve()
    canonical = (repo / "evidence" / f"ACL-005-confirmatory-{approved_sha}.json").resolve()
    if requested != canonical:
        raise ValueError("ACL-005 output must equal the SHA-derived canonical evidence path")
    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = repo / bundle
    bundle = bundle.resolve()
    canonical_bundle = (repo / "preregistrations" / "ACL-005").resolve()
    if bundle != canonical_bundle:
        raise ValueError("ACL-005 requires the SHA-bound canonical preregistration bundle")
    current_sha, dirty = git_execution_state(repo)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        worktree_dirty=dirty,
        output_path=canonical,
    )
    validation = validate_preregistration_bundle(bundle)
    manifest = load_manifest(bundle / "manifest.json")
    source_validation = validate_source_evidence(repo, manifest)
    locked_registry = json.loads(
        (bundle / "analytic_registry.json").read_text(encoding="utf-8")
    )
    locked_bundle = json.loads((bundle / "LOCK.json").read_text(encoding="utf-8"))
    results = [estimate_landscape(manifest, landscape) for landscape in manifest.landscapes]
    for result in results:
        reproduced = reproduce_stopped_mean(result)
        if not np.array_equal(reproduced, np.asarray(result["stopped_mean_direction"])):
            raise FloatingPointError("ACL-005 chunks do not reproduce stopped mean")
    analysis = analyze_landscape_results(manifest, results)
    payload = {
        "schema_version": 1,
        "experiment_id": "ACL-005",
        "kind": "confirmatory-cross-class-contextual-bandit-conditional-mean",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "source_evidence_validation": source_validation,
        "randomness": "PCG64 independent landscape streams",
        "target_refit": False,
        "source_experiment": manifest.source_experiment,
        "source_rule": manifest.source_rule,
        "benchmark_scope": manifest.benchmark_scope,
        "inference_scope": manifest.inference_scope,
        "transport_scope": manifest.transport_scope,
        "frozen_design": manifest.raw,
        "locked_analytic_registry": locked_registry,
        "preregistration_bundle_lock": locked_bundle,
        "landscape_results": results,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(canonical, payload)
