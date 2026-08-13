"""ACL-007 no-refit transport into sequential Bayesian particle filtering."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acl002 import (
    assert_execution_context,
    git_execution_state,
    sha256_file,
    type7_quantile,
    validate_lock,
)
from .io import provenance, write_json
from .particle_filter_bias import (
    ParticleFilterSpecification,
    exact_particle_filter_moments,
    simulate_labeled_particle_filters,
)

FloatArray = NDArray[np.float64]

ACL007_STATES = 3
ACL007_REPLICATION_SCHEDULE = (8192, 32768, 131072, 262144)
ACL007_CHUNK_SIZE = 4096
ACL007_DIRECTION_SCORE_MAX = 5.0
ACL007_FULL_SCORE_MEDIAN_MAX = 1.5
ACL007_FULL_SCORE_Q90_MAX = 2.5
ACL007_DISSOCIATION_EXACT_TRUTH_MAX = 0.90
ACL007_DISSOCIATION_OBSERVED_TRUTH_MAX = 0.95
ACL007_DISSOCIATION_HALF_COSINE_MIN = 0.995
ACL007_RESOLVABLE_CONTRAST_GAP_MIN = 0.10
ACL007_REGISTRY_ATOL = 2e-12
ACL007_REGISTRY_RTOL = 2e-12
ACL007_SOURCE_EXPERIMENT = "ACL-006"
ACL007_SOURCE_RULE = "unchanged-native-metric-standardized-mean-and-dissociation-law"
ACL007_SOURCE_EVIDENCE = {
    "approved_preregistration_sha": "a8b42042e397f1422866a0ca9496ee07abe0a42a",
    "evidence_commit": "c94890dc8f361c0309802c0ef0173ec84e814d3d",
    "evidence_artifact": (
        "evidence/ACL-006-confirmatory-"
        "a8b42042e397f1422866a0ca9496ee07abe0a42a.json"
    ),
    "evidence_sha256": "740c541bbd69db77f6d02327ded34765a37345f907b048f8d3f3a91aebc23918",
    "report_commit": "c8d599fe09887e22fe02f92a27bcc8c13ac4baf4",
    "report_summary": "analysis/ACL-006-confirmatory/summary.json",
    "report_summary_sha256": (
        "0748482b3796b861267fdb5781bab11605cfc82263e4a6fdbd206df4b96acd6c"
    ),
    "source_exact_mean_verdict": "PASS",
    "source_dissociation_verdict": "PASS",
    "source_contrast_verdict": "PASS",
    "source_direction_score_maximum": 1.7073805833110742,
    "source_full_score_median_type7": 0.6095572205053943,
    "source_full_score_q90_type7": 1.272615265047392,
}
ACL007_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.13.14",
    "numpy_version": "2.5.2",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
}
ACL007_TARGET_IDS = tuple(
    [f"A{index:02d}" for index in range(1, 13)]
    + [f"B{index:02d}" for index in range(1, 5)]
)
ACL007_CONTRAST_IDS = (
    "A-correct-vs-reversed-N3",
    "A-correct-vs-reversed-N4",
    "A-correct-vs-reversed-N6",
    "A-correct-vs-reversed-N8",
    "A-correct-vs-flat-N4",
    "A-correct-vs-flat-N8",
    "A-correct-vs-missing-N4",
    "A-correct-vs-missing-N8",
    "B-correct-vs-reversed-N4",
)
ACL007_DISSOCIATION_IDS = ("A06", "A07", "A08")
ACL007_RESOLVABLE_CONTRAST_IDS = ACL007_CONTRAST_IDS
ACL007_LOCKED_FILES = frozenset(
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
class ACL007Target:
    identifier: str
    family: str
    model_identifier: str
    particle_count: int
    specification: ParticleFilterSpecification
    seed: int


@dataclass(frozen=True)
class ACL007Contrast:
    identifier: str
    kind: str
    left: str
    right: str


@dataclass(frozen=True)
class ACL007Manifest:
    experiment_id: str
    states: int
    replication_schedule: tuple[int, ...]
    chunk_size: int
    direction_score_max: float
    full_score_median_max: float
    full_score_q90_max: float
    dissociation_exact_truth_cosine_max: float
    dissociation_observed_truth_cosine_max: float
    dissociation_half_cosine_min: float
    resolvable_contrast_gap_min: float
    analytic_registry_atol: float
    analytic_registry_rtol: float
    source_experiment: str
    source_rule: str
    targets: tuple[ACL007Target, ...]
    contrasts: tuple[ACL007Contrast, ...]
    benchmark_scope: str
    inference_scope: str
    transport_scope: str
    confirmatory_environment: dict[str, str]
    raw: dict[str, Any]


def _integer(payload: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _positive_float(payload: dict[str, Any], key: str) -> float:
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be finite and positive") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _cosine_threshold(payload: dict[str, Any], key: str) -> float:
    value = _positive_float(payload, key)
    if value > 1.0:
        raise ValueError(f"{key} cannot exceed one")
    return value


def validate_manifest_dict(payload: dict[str, Any]) -> ACL007Manifest:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ACL-007 manifest schema")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if payload.get("randomness") != "PCG64-independent-target-streams":
        raise ValueError("ACL-007 randomness mismatch")
    states = _integer(payload, "states", minimum=2)
    schedule_payload = payload.get("replication_schedule")
    if not isinstance(schedule_payload, list) or not schedule_payload:
        raise ValueError("replication_schedule must be non-empty")
    schedule = tuple(schedule_payload)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in schedule):
        raise ValueError("replication_schedule must contain integers")
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError("replication_schedule must be strictly increasing")
    chunk_size = _integer(payload, "shadow_chunk_size")
    if any(value % (2 * chunk_size) != 0 for value in schedule):
        raise ValueError("every checkpoint and half must align to complete chunks")
    direction_score_max = _positive_float(payload, "direction_score_max")
    median_max = _positive_float(payload, "full_score_median_max")
    q90_max = _positive_float(payload, "full_score_q90_max")
    dissociation_exact = _cosine_threshold(
        payload, "dissociation_exact_truth_cosine_max"
    )
    dissociation_observed = _cosine_threshold(
        payload, "dissociation_observed_truth_cosine_max"
    )
    dissociation_half = _cosine_threshold(payload, "dissociation_half_cosine_min")
    if dissociation_exact >= dissociation_observed:
        raise ValueError("dissociation exact threshold must be below observed threshold")
    contrast_gap = _positive_float(payload, "resolvable_contrast_gap_min")
    registry_atol = _positive_float(payload, "analytic_registry_atol")
    registry_rtol = _positive_float(payload, "analytic_registry_rtol")
    source_experiment = payload.get("source_experiment")
    source_rule = payload.get("source_rule")
    if not isinstance(source_experiment, str) or not source_experiment:
        raise ValueError("source_experiment must be non-empty")
    if not isinstance(source_rule, str) or not source_rule:
        raise ValueError("source_rule must be non-empty")

    model_payload = payload.get("models")
    if not isinstance(model_payload, dict) or not model_payload:
        raise ValueError("models must be a non-empty object")
    true_models: dict[str, dict[str, Any]] = {}
    for identifier, model in model_payload.items():
        if not isinstance(identifier, str) or not identifier or not isinstance(model, dict):
            raise ValueError("model IDs and payloads must be valid")
        true_models[identifier] = model

    target_payload = payload.get("targets")
    if not isinstance(target_payload, list) or not target_payload:
        raise ValueError("targets must be a non-empty list")
    targets: list[ACL007Target] = []
    identifiers: set[str] = set()
    seeds: set[int] = set()
    for item in target_payload:
        if not isinstance(item, dict):
            raise ValueError("each ACL-007 target must be an object")
        identifier = item.get("id")
        family = item.get("family")
        model_identifier = item.get("model")
        seed = item.get("seed")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("target IDs must be non-empty and unique")
        if not isinstance(family, str) or not family:
            raise ValueError("target family must be non-empty")
        if model_identifier not in true_models:
            raise ValueError("target model must name a frozen true model")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in seeds:
            raise ValueError("target seeds must be distinct non-negative integers")
        particle_count = _integer(item, "particle_count")
        model = true_models[model_identifier]
        specification = ParticleFilterSpecification(
            initial_belief=model.get("initial_belief"),
            true_transition=model.get("true_transition"),
            filter_transition=item.get("filter_transition"),
            true_likelihoods=model.get("true_likelihoods"),
            filter_likelihoods=item.get("filter_likelihoods"),
        )
        initial, _, _, _, _ = specification.arrays()
        if initial.size != states:
            raise ValueError("target state count differs from manifest")
        targets.append(
            ACL007Target(
                identifier,
                family,
                model_identifier,
                particle_count,
                specification,
                seed,
            )
        )
        identifiers.add(identifier)
        seeds.add(seed)

    contrast_payload = payload.get("contrasts")
    if not isinstance(contrast_payload, list) or not contrast_payload:
        raise ValueError("contrasts must be a non-empty list")
    contrasts: list[ACL007Contrast] = []
    contrast_ids: set[str] = set()
    target_by_id = {target.identifier: target for target in targets}
    for item in contrast_payload:
        if not isinstance(item, dict):
            raise ValueError("each ACL-007 contrast must be an object")
        identifier = item.get("id")
        kind = item.get("kind")
        left = item.get("left")
        right = item.get("right")
        if not isinstance(identifier, str) or not identifier or identifier in contrast_ids:
            raise ValueError("contrast IDs must be non-empty and unique")
        if kind not in {"observation-misspecification", "missing-observation"}:
            raise ValueError("unsupported ACL-007 contrast kind")
        if left not in target_by_id or right not in target_by_id or left == right:
            raise ValueError("contrast endpoints must be distinct frozen targets")
        left_target = target_by_id[left]
        right_target = target_by_id[right]
        if (
            left_target.model_identifier != right_target.model_identifier
            or left_target.particle_count != right_target.particle_count
        ):
            raise ValueError("ACL-007 contrasts must share true model and particle count")
        contrasts.append(ACL007Contrast(identifier, kind, left, right))
        contrast_ids.add(identifier)

    scopes = tuple(
        payload.get(key)
        for key in ("benchmark_scope", "inference_scope", "transport_scope")
    )
    if not all(isinstance(value, str) and value for value in scopes):
        raise ValueError("ACL-007 scope fields must be non-empty")
    environment = payload.get("confirmatory_environment")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("confirmatory_environment must be a string mapping")

    if experiment_id == "ACL-007":
        exact = (
            states == ACL007_STATES
            and schedule == ACL007_REPLICATION_SCHEDULE
            and chunk_size == ACL007_CHUNK_SIZE
            and direction_score_max == ACL007_DIRECTION_SCORE_MAX
            and median_max == ACL007_FULL_SCORE_MEDIAN_MAX
            and q90_max == ACL007_FULL_SCORE_Q90_MAX
            and dissociation_exact == ACL007_DISSOCIATION_EXACT_TRUTH_MAX
            and dissociation_observed == ACL007_DISSOCIATION_OBSERVED_TRUTH_MAX
            and dissociation_half == ACL007_DISSOCIATION_HALF_COSINE_MIN
            and contrast_gap == ACL007_RESOLVABLE_CONTRAST_GAP_MIN
            and registry_atol == ACL007_REGISTRY_ATOL
            and registry_rtol == ACL007_REGISTRY_RTOL
            and source_experiment == ACL007_SOURCE_EXPERIMENT
            and source_rule == ACL007_SOURCE_RULE
            and payload.get("source_evidence") == ACL007_SOURCE_EVIDENCE
            and tuple(target.identifier for target in targets) == ACL007_TARGET_IDS
            and tuple(contrast.identifier for contrast in contrasts)
            == ACL007_CONTRAST_IDS
            and environment == ACL007_ENVIRONMENT
            and scopes
            == (
                "deterministic-held-out-sequential-inference-benchmark",
                "descriptive-criteria-not-population-confidence",
                "ACL-006-to-particle-filter-unchanged-native-metric-diagnostic",
            )
        )
        if not exact:
            raise ValueError("ACL-007 design constants mismatch")

    return ACL007Manifest(
        experiment_id=experiment_id,
        states=states,
        replication_schedule=schedule,
        chunk_size=chunk_size,
        direction_score_max=direction_score_max,
        full_score_median_max=median_max,
        full_score_q90_max=q90_max,
        dissociation_exact_truth_cosine_max=dissociation_exact,
        dissociation_observed_truth_cosine_max=dissociation_observed,
        dissociation_half_cosine_min=dissociation_half,
        resolvable_contrast_gap_min=contrast_gap,
        analytic_registry_atol=registry_atol,
        analytic_registry_rtol=registry_rtol,
        source_experiment=source_experiment,
        source_rule=source_rule,
        targets=tuple(targets),
        contrasts=tuple(contrasts),
        benchmark_scope=scopes[0],
        inference_scope=scopes[1],
        transport_scope=scopes[2],
        confirmatory_environment=dict(environment),
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL007Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-007 manifest") from error
    return validate_manifest_dict(payload)


def _cosine(left: ArrayLike, right: ArrayLike) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    numerator = float(left_array @ right_array)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Euclidean cosine requires nonzero finite directions")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _angular_envelope(mean_norm: float, rms_se: float, score_max: float) -> float:
    radius = score_max * rms_se
    if radius >= mean_norm:
        return 2.0
    return float(min(2.0, 2.0 * radius / (mean_norm - radius)))


def _half_cosine_lower_bound(mean_norm: float, half_se: float, score_max: float) -> float:
    radius = score_max * half_se
    if radius >= mean_norm:
        return -1.0
    unit_error = 2.0 * radius / (mean_norm - radius)
    return float(max(-1.0, 1.0 - 2.0 * unit_error**2))


def build_analytic_registry(manifest: ACL007Manifest) -> dict[str, Any]:
    final_replications = manifest.replication_schedule[-1]
    entries = []
    by_id: dict[str, dict[str, Any]] = {}
    for target in manifest.targets:
        moments = exact_particle_filter_moments(
            target.specification, particle_count=target.particle_count
        )
        initial, true_transition, filter_transition, true_likelihoods, filter_likelihoods = (
            target.specification.arrays()
        )
        mean_norm = float(np.linalg.norm(moments.mean_update))
        analytic_norm = float(np.linalg.norm(moments.analytic_update))
        variance_trace = float(np.trace(moments.covariance))
        if min(mean_norm, analytic_norm, variance_trace) <= 1e-12:
            raise FloatingPointError("ACL-007 direction/variance scale is too small")
        full_se = float(np.sqrt(variance_trace / final_replications))
        half_se = float(np.sqrt(variance_trace / (final_replications / 2)))
        envelope = _angular_envelope(
            mean_norm, full_se, manifest.direction_score_max
        )
        half_lower = _half_cosine_lower_bound(
            mean_norm, half_se, manifest.direction_score_max
        )
        exact_cosine = moments.truth_alignment_cosine
        dissociation = (
            exact_cosine <= manifest.dissociation_exact_truth_cosine_max
            and exact_cosine + envelope
            <= manifest.dissociation_observed_truth_cosine_max
            and half_lower >= manifest.dissociation_half_cosine_min
        )
        entry = {
            "id": target.identifier,
            "family": target.family,
            "model": target.model_identifier,
            "seed": target.seed,
            "particle_count": target.particle_count,
            "initial_belief": initial.tolist(),
            "true_transition": true_transition.tolist(),
            "filter_transition": filter_transition.tolist(),
            "true_likelihoods": true_likelihoods.tolist(),
            "filter_likelihoods": filter_likelihoods.tolist(),
            "exact_true_belief_trajectory": moments.exact_belief_trajectory.tolist(),
            "analytic_update": moments.analytic_update.tolist(),
            "exact_mean_belief": moments.mean_belief.tolist(),
            "exact_mean_direction": moments.mean_update.tolist(),
            "exact_direction_covariance": moments.covariance.tolist(),
            "exact_probability_mass": moments.probability_mass,
            "exact_truth_alignment_cosine": exact_cosine,
            "exact_truth_angular_loss": 1.0 - exact_cosine,
            "terminal_missing_state_probability": (
                moments.terminal_missing_state_probability.tolist()
            ),
            "terminal_support_size_probabilities": {
                str(key): value
                for key, value in moments.terminal_support_size_probabilities.items()
            },
            "exact_mean_direction_norm": mean_norm,
            "analytic_direction_norm": analytic_norm,
            "single_shadow_variance_trace": variance_trace,
            "final_replications": final_replications,
            "final_rms_standard_error": full_se,
            "half_rms_standard_error": half_se,
            "final_angular_envelope": envelope,
            "two_half_cosine_lower_bound": half_lower,
            "dissociation_stratum": dissociation,
        }
        entries.append(entry)
        by_id[target.identifier] = entry

    contrast_entries = []
    for contrast in manifest.contrasts:
        left = by_id[contrast.left]
        right = by_id[contrast.right]
        signed_gap = (
            right["exact_truth_alignment_cosine"]
            - left["exact_truth_alignment_cosine"]
        )
        conservative = (
            abs(signed_gap)
            - left["final_angular_envelope"]
            - right["final_angular_envelope"]
        )
        contrast_entries.append(
            {
                "id": contrast.identifier,
                "kind": contrast.kind,
                "left": contrast.left,
                "right": contrast.right,
                "exact_signed_truth_cosine_gap": signed_gap,
                "conservative_absolute_gap": conservative,
                "resolvable": conservative >= manifest.resolvable_contrast_gap_min,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "exact-finite-particle-filter-registry",
        "outcomes_generated": False,
        "shadow_count": 0,
        "target_refit": False,
        "native_metric": "centered-euclidean-belief-tangent",
        "comparator_source": "exact-true-model-Bayes-filter-and-count-state-PF-law",
        "transported_source_rule": manifest.source_rule,
        "numeric_comparison": {
            "absolute_tolerance": manifest.analytic_registry_atol,
            "relative_tolerance": manifest.analytic_registry_rtol,
        },
        "targets": entries,
        "contrasts": contrast_entries,
        "dissociation_target_ids": [
            entry["id"] for entry in entries if entry["dissociation_stratum"]
        ],
        "resolvable_contrast_ids": [
            entry["id"] for entry in contrast_entries if entry["resolvable"]
        ],
    }


def _assert_numeric_equivalence(
    locked: Any,
    recomputed: Any,
    *,
    atol: float,
    rtol: float,
    path: str = "$",
) -> None:
    if isinstance(locked, bool) or isinstance(recomputed, bool):
        if locked is not recomputed:
            raise ValueError(f"analytic registry differs at {path}")
        return
    if isinstance(locked, int) and isinstance(recomputed, int):
        if locked != recomputed:
            raise ValueError(f"analytic registry differs at {path}")
        return
    if isinstance(locked, (int, float)) and isinstance(recomputed, (int, float)):
        if not np.isclose(float(locked), float(recomputed), atol=atol, rtol=rtol):
            raise ValueError(f"analytic registry differs numerically at {path}")
        return
    if isinstance(locked, str) and isinstance(recomputed, str):
        if locked != recomputed:
            raise ValueError(f"analytic registry differs at {path}")
        return
    if locked is None or recomputed is None:
        if locked is not recomputed:
            raise ValueError(f"analytic registry differs at {path}")
        return
    if isinstance(locked, list) and isinstance(recomputed, list):
        if len(locked) != len(recomputed):
            raise ValueError(f"analytic registry length differs at {path}")
        for index, (left, right) in enumerate(zip(locked, recomputed, strict=True)):
            _assert_numeric_equivalence(
                left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]"
            )
        return
    if isinstance(locked, dict) and isinstance(recomputed, dict):
        if set(locked) != set(recomputed):
            raise ValueError(f"analytic registry keys differ at {path}")
        for key in sorted(locked):
            _assert_numeric_equivalence(
                locked[key],
                recomputed[key],
                atol=atol,
                rtol=rtol,
                path=f"{path}.{key}",
            )
        return
    raise ValueError(f"analytic registry types differ at {path}")


def reproduce_target_mean(result: dict[str, Any]) -> FloatArray:
    chunks = result.get("shadow_chunks", [])
    if not chunks:
        raise ValueError("target result has no shadow chunks")
    count = sum(int(chunk["count"]) for chunk in chunks)
    total = np.sum(
        np.asarray([chunk["direction_sum"] for chunk in chunks], dtype=np.float64),
        axis=0,
    )
    return total / count


def _mean_from_chunks(chunks: list[dict[str, Any]]) -> FloatArray:
    return reproduce_target_mean({"shadow_chunks": chunks})


def _direction_score(
    observed: FloatArray, exact: FloatArray, rms_standard_error: float
) -> float:
    if not np.isfinite(rms_standard_error) or rms_standard_error <= 0.0:
        raise FloatingPointError("direction score scale must be positive")
    return float(np.linalg.norm(observed - exact) / rms_standard_error)


def estimate_target(
    manifest: ACL007Manifest,
    target: ACL007Target,
    registry_entry: dict[str, Any],
) -> dict[str, Any]:
    exact_mean = np.asarray(registry_entry["exact_mean_direction"], dtype=np.float64)
    analytic = np.asarray(registry_entry["analytic_update"], dtype=np.float64)
    initial = np.asarray(registry_entry["initial_belief"], dtype=np.float64)
    variance_trace = float(registry_entry["single_shadow_variance_trace"])
    rng = np.random.Generator(np.random.PCG64(target.seed))
    chunks: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints = set(manifest.replication_schedule)
    generated = 0
    while generated < manifest.replication_schedule[-1]:
        beliefs, particles = simulate_labeled_particle_filters(
            target.specification,
            particle_count=target.particle_count,
            replications=manifest.chunk_size,
            rng=rng,
            batch_size=manifest.chunk_size,
        )
        directions = beliefs - initial[None, :]
        masks = np.sum(
            np.any(
                particles[:, :, None]
                == np.arange(manifest.states, dtype=np.int64)[None, None, :],
                axis=1,
            ).astype(np.int64)
            * (1 << np.arange(manifest.states, dtype=np.int64))[None, :],
            axis=1,
        )
        chunks.append(
            {
                "chunk_index": len(chunks),
                "count": manifest.chunk_size,
                "direction_sum": np.sum(directions, axis=0).tolist(),
                "direction_outer_sum": (directions.T @ directions).tolist(),
                "terminal_support_mask_counts": np.bincount(
                    masks, minlength=1 << manifest.states
                ).tolist(),
            }
        )
        generated += manifest.chunk_size
        if generated not in checkpoints:
            continue
        prefix = chunks
        midpoint = len(prefix) // 2
        full_mean = _mean_from_chunks(prefix)
        first_mean = _mean_from_chunks(prefix[:midpoint])
        second_mean = _mean_from_chunks(prefix[midpoint:])
        full_se = float(np.sqrt(variance_trace / generated))
        half_se = float(np.sqrt(variance_trace / (generated / 2)))
        history.append(
            {
                "replications": generated,
                "full_mean_direction": full_mean.tolist(),
                "first_half_mean_direction": first_mean.tolist(),
                "second_half_mean_direction": second_mean.tolist(),
                "full_direction_score": _direction_score(full_mean, exact_mean, full_se),
                "first_half_direction_score": _direction_score(
                    first_mean, exact_mean, half_se
                ),
                "second_half_direction_score": _direction_score(
                    second_mean, exact_mean, half_se
                ),
                "full_truth_cosine": _cosine(full_mean, analytic),
                "half_cosine": _cosine(first_mean, second_mean),
            }
        )
    final = history[-1]
    observed_truth = float(final["full_truth_cosine"])
    exact_truth = float(registry_entry["exact_truth_alignment_cosine"])
    return {
        "target_id": target.identifier,
        "family": target.family,
        "model": target.model_identifier,
        "seed": target.seed,
        "particle_count": target.particle_count,
        "generated_replications": generated,
        "exact_mean_direction": exact_mean.tolist(),
        "observed_mean_direction": final["full_mean_direction"],
        "full_direction_score": final["full_direction_score"],
        "first_half_direction_score": final["first_half_direction_score"],
        "second_half_direction_score": final["second_half_direction_score"],
        "exact_truth_cosine": exact_truth,
        "observed_truth_cosine": observed_truth,
        "angular_residual": abs(observed_truth - exact_truth),
        "angular_envelope": registry_entry["final_angular_envelope"],
        "final_half_cosine": final["half_cosine"],
        "dissociation_stratum": registry_entry["dissociation_stratum"],
        "checkpoint_history": history,
        "shadow_chunks": chunks,
        "rng_state_after": rng.bit_generator.state,
    }


def validate_target_result(
    manifest: ACL007Manifest,
    target: ACL007Target,
    registry_entry: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if (
        result.get("target_id") != target.identifier
        or result.get("family") != target.family
        or result.get("model") != target.model_identifier
        or result.get("seed") != target.seed
        or result.get("particle_count") != target.particle_count
    ):
        raise ValueError("ACL-007 target identity differs from manifest")
    chunks = result.get("shadow_chunks")
    history = result.get("checkpoint_history")
    if not isinstance(chunks, list) or not isinstance(history, list):
        raise ValueError("ACL-007 target lacks chunks/checkpoints")
    if len(chunks) != manifest.replication_schedule[-1] // manifest.chunk_size:
        raise ValueError("ACL-007 target has the wrong fixed budget")
    if len(history) != len(manifest.replication_schedule):
        raise ValueError("ACL-007 target has the wrong checkpoint count")
    for index, chunk in enumerate(chunks):
        direction_sum = np.asarray(chunk.get("direction_sum"), dtype=np.float64)
        support = np.asarray(chunk.get("terminal_support_mask_counts"), dtype=np.int64)
        outer = np.asarray(chunk.get("direction_outer_sum"), dtype=np.float64)
        if (
            chunk.get("chunk_index") != index
            or chunk.get("count") != manifest.chunk_size
            or direction_sum.shape != (manifest.states,)
            or not np.all(np.isfinite(direction_sum))
            or support.shape != (1 << manifest.states,)
            or np.any(support < 0)
            or int(np.sum(support)) != manifest.chunk_size
            or outer.shape != (manifest.states, manifest.states)
            or not np.all(np.isfinite(outer))
            or not np.array_equal(outer, outer.T)
        ):
            raise ValueError("ACL-007 chunk sufficient statistics are invalid")
    exact = np.asarray(registry_entry["exact_mean_direction"], dtype=np.float64)
    analytic = np.asarray(registry_entry["analytic_update"], dtype=np.float64)
    variance = float(registry_entry["single_shadow_variance_trace"])
    for checkpoint_count, checkpoint in zip(
        manifest.replication_schedule, history, strict=True
    ):
        prefix = chunks[: checkpoint_count // manifest.chunk_size]
        midpoint = len(prefix) // 2
        means = (
            _mean_from_chunks(prefix),
            _mean_from_chunks(prefix[:midpoint]),
            _mean_from_chunks(prefix[midpoint:]),
        )
        stored_names = (
            "full_mean_direction",
            "first_half_mean_direction",
            "second_half_mean_direction",
        )
        if checkpoint.get("replications") != checkpoint_count:
            raise ValueError("ACL-007 checkpoint schedule differs")
        for name, mean in zip(stored_names, means, strict=True):
            if not np.array_equal(np.asarray(checkpoint.get(name)), mean):
                raise FloatingPointError(f"ACL-007 chunks do not reproduce {name}")
        full_se = float(np.sqrt(variance / checkpoint_count))
        half_se = float(np.sqrt(variance / (checkpoint_count / 2)))
        scalars = {
            "full_direction_score": _direction_score(means[0], exact, full_se),
            "first_half_direction_score": _direction_score(means[1], exact, half_se),
            "second_half_direction_score": _direction_score(means[2], exact, half_se),
            "full_truth_cosine": _cosine(means[0], analytic),
            "half_cosine": _cosine(means[1], means[2]),
        }
        for name, expected in scalars.items():
            if not np.isclose(
                float(checkpoint.get(name)), expected, atol=5e-14, rtol=5e-14
            ):
                raise FloatingPointError(f"ACL-007 chunks do not reproduce {name}")
    final = history[-1]
    if result.get("generated_replications") != manifest.replication_schedule[-1]:
        raise ValueError("ACL-007 final replication count differs")
    if not np.array_equal(
        np.asarray(result.get("exact_mean_direction"), dtype=np.float64), exact
    ):
        raise FloatingPointError("ACL-007 final exact mean differs from registry")
    if not np.array_equal(
        np.asarray(result.get("observed_mean_direction"), dtype=np.float64),
        np.asarray(final["full_mean_direction"], dtype=np.float64),
    ):
        raise FloatingPointError("ACL-007 final mean differs from checkpoint")
    final_fields = {
        "full_direction_score": float(final["full_direction_score"]),
        "first_half_direction_score": float(final["first_half_direction_score"]),
        "second_half_direction_score": float(final["second_half_direction_score"]),
        "observed_truth_cosine": float(final["full_truth_cosine"]),
        "final_half_cosine": float(final["half_cosine"]),
    }
    for name, expected in final_fields.items():
        if float(result.get(name)) != expected:
            raise FloatingPointError(f"ACL-007 final result differs for {name}")
    exact_truth = float(registry_entry["exact_truth_alignment_cosine"])
    if float(result.get("exact_truth_cosine")) != exact_truth or not np.isclose(
        float(result.get("angular_residual")),
        abs(final_fields["observed_truth_cosine"] - exact_truth),
        atol=5e-16,
        rtol=5e-16,
    ):
        raise FloatingPointError("ACL-007 final angular residual is invalid")
    if (
        float(result.get("angular_envelope"))
        != float(registry_entry["final_angular_envelope"])
        or bool(result.get("dissociation_stratum"))
        is not bool(registry_entry["dissociation_stratum"])
    ):
        raise ValueError("ACL-007 final analytic labels differ from registry")
    rng_state = result.get("rng_state_after")
    if not isinstance(rng_state, dict) or rng_state.get("bit_generator") != "PCG64":
        raise ValueError("ACL-007 final RNG state is missing or invalid")


def analyze_target_results(
    manifest: ACL007Manifest,
    registry: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_ids = [target.identifier for target in manifest.targets]
    if [row.get("target_id") for row in results] != expected_ids:
        raise ValueError("ACL-007 result target IDs or order differ from manifest")
    registry_by_id = {entry["id"]: entry for entry in registry["targets"]}
    rows_by_id = {row["target_id"]: row for row in results}
    keys = (
        "full_direction_score",
        "first_half_direction_score",
        "second_half_direction_score",
        "angular_residual",
        "angular_envelope",
        "observed_truth_cosine",
        "final_half_cosine",
    )
    if any(not np.isfinite(float(row[key])) for row in results for key in keys):
        raise FloatingPointError("ACL-007 analysis received non-finite results")
    full_scores = [float(row["full_direction_score"]) for row in results]
    all_scores = [
        float(row[key])
        for row in results
        for key in (
            "full_direction_score",
            "first_half_direction_score",
            "second_half_direction_score",
        )
    ]
    median = type7_quantile(full_scores, 0.5)
    q90 = type7_quantile(full_scores, 0.9)
    angular_ok = all(
        float(row["angular_residual"])
        <= float(registry_by_id[row["target_id"]]["final_angular_envelope"])
        and float(row["angular_envelope"])
        == float(registry_by_id[row["target_id"]]["final_angular_envelope"])
        for row in results
    )
    mean_pass = (
        max(all_scores) <= manifest.direction_score_max
        and median <= manifest.full_score_median_max
        and q90 <= manifest.full_score_q90_max
        and angular_ok
    )
    dissociation = [
        row
        for row in results
        if registry_by_id[row["target_id"]]["dissociation_stratum"]
    ]
    if not dissociation:
        dissociation_verdict = "NOT_APPLICABLE"
    elif all(
        float(row["final_half_cosine"]) >= manifest.dissociation_half_cosine_min
        and float(row["observed_truth_cosine"])
        <= manifest.dissociation_observed_truth_cosine_max
        for row in dissociation
    ):
        dissociation_verdict = "PASS"
    else:
        dissociation_verdict = "FAIL"
    contrast_rows = []
    gating = []
    for contrast in registry["contrasts"]:
        observed = (
            float(rows_by_id[contrast["right"]]["observed_truth_cosine"])
            - float(rows_by_id[contrast["left"]]["observed_truth_cosine"])
        )
        predicted = float(contrast["exact_signed_truth_cosine_gap"])
        reproduced = (
            observed * predicted > 0.0
            and abs(observed) >= manifest.resolvable_contrast_gap_min
        )
        if contrast["resolvable"]:
            gating.append(reproduced)
        contrast_rows.append(
            {
                **contrast,
                "observed_signed_truth_cosine_gap": observed,
                "predicted_sign_reproduced": observed * predicted > 0.0,
                "observed_gap_retains_minimum": (
                    abs(observed) >= manifest.resolvable_contrast_gap_min
                ),
                "gating_reproduced": reproduced if contrast["resolvable"] else None,
            }
        )
    contrast_verdict = (
        "NOT_APPLICABLE" if not gating else "PASS" if all(gating) else "FAIL"
    )
    if manifest.experiment_id == "ACL-007" and not dissociation:
        transport_verdict = "INVALID"
    elif (
        mean_pass
        and dissociation_verdict == "PASS"
        and contrast_verdict == "PASS"
    ):
        transport_verdict = "PASS"
    else:
        transport_verdict = "FAIL"
    return {
        "experiment_id": manifest.experiment_id,
        "transport_verdict": transport_verdict,
        "standardized_mean_prediction_verdict": "PASS" if mean_pass else "FAIL",
        "dissociation_prediction_verdict": dissociation_verdict,
        "contrast_reproduction_verdict": contrast_verdict,
        "direction_score_maximum": max(all_scores),
        "full_direction_score_median_type7": median,
        "full_direction_score_q90_type7": q90,
        "all_angular_residuals_within_frozen_envelopes": angular_ok,
        "dissociation_target_ids": [row["target_id"] for row in dissociation],
        "target_refit": False,
        "source_thresholds_changed": False,
        "native_metric_changed_as_preregistered": True,
        "population_confidence_claim": False,
        "contrast_results": contrast_rows,
    }


def validate_execution_environment(manifest: ACL007Manifest) -> dict[str, Any]:
    actual = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    if actual != manifest.confirmatory_environment:
        raise ValueError("ACL-007 confirmatory environment mismatch")
    return {"valid": True, **actual, "dtype": "float64", "rng": "PCG64"}


def validate_source_evidence(
    repo_path: str | Path, manifest: ACL007Manifest
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    source = manifest.raw.get("source_evidence")
    if source != ACL007_SOURCE_EVIDENCE:
        raise ValueError("ACL-007 source evidence differs from frozen source record")
    evidence = (repo / source["evidence_artifact"]).resolve()
    report = (repo / source["report_summary"]).resolve()
    try:
        evidence.relative_to(repo)
        report.relative_to(repo)
        evidence_hash = sha256_file(evidence)
        report_hash = sha256_file(report)
    except (OSError, ValueError) as error:
        raise ValueError("cannot read frozen ACL-006 source evidence") from error
    if evidence_hash != source["evidence_sha256"]:
        raise ValueError("ACL-006 source evidence SHA-256 mismatch")
    if report_hash != source["report_summary_sha256"]:
        raise ValueError("ACL-006 source report SHA-256 mismatch")
    return {
        "valid": True,
        "source_experiment": manifest.source_experiment,
        "evidence_artifact": source["evidence_artifact"],
        "evidence_sha256": source["evidence_sha256"],
        "report_summary": source["report_summary"],
        "report_summary_sha256": source["report_summary_sha256"],
    }


def validate_preregistration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if (
        lock.get("experiment_id") != "ACL-007"
        or lock.get("kind") != "preregistration-bundle-lock"
        or lock.get("outcomes_generated") is not False
        or set(lock.get("files", {})) != ACL007_LOCKED_FILES
    ):
        raise ValueError("ACL-007 lock must contain exact frozen file set")
    actual_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if actual_files != ACL007_LOCKED_FILES | {"LOCK.json"}:
        raise ValueError("ACL-007 bundle must have exact frozen directory contents")
    manifest = load_manifest(bundle / "manifest.json")
    try:
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-007 analytic registry") from error
    recomputed = build_analytic_registry(manifest)
    _assert_numeric_equivalence(
        locked_registry,
        recomputed,
        atol=manifest.analytic_registry_atol,
        rtol=manifest.analytic_registry_rtol,
    )
    if locked_registry.get("outcomes_generated") is not False:
        raise ValueError("ACL-007 registry must remain analytic-only")
    if manifest.experiment_id == "ACL-007" and not locked_registry.get(
        "dissociation_target_ids"
    ):
        raise ValueError("ACL-007 frozen dissociation stratum must be nonempty")
    if manifest.experiment_id == "ACL-007" and (
        tuple(locked_registry["dissociation_target_ids"])
        != ACL007_DISSOCIATION_IDS
        or tuple(locked_registry["resolvable_contrast_ids"])
        != ACL007_RESOLVABLE_CONTRAST_IDS
    ):
        raise ValueError("ACL-007 analytic strata differ from frozen design")
    return {
        "schema_version": 1,
        "experiment_id": "ACL-007",
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "target_count": len(manifest.targets),
        "dissociation_target_count": len(locked_registry["dissociation_target_ids"]),
        "resolvable_contrast_count": len(locked_registry["resolvable_contrast_ids"]),
        "registry_comparison": "numeric-tolerance",
        "locked_file_count": len(lock["files"]),
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
    canonical = (repo / "evidence" / f"ACL-007-confirmatory-{approved_sha}.json").resolve()
    if requested != canonical:
        raise ValueError("ACL-007 output must equal the SHA-derived canonical evidence path")
    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = repo / bundle
    bundle = bundle.resolve()
    if bundle != (repo / "preregistrations" / "ACL-007").resolve():
        raise ValueError("ACL-007 requires the SHA-bound canonical preregistration bundle")
    current_sha, dirty = git_execution_state(repo)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        worktree_dirty=dirty,
        output_path=canonical,
    )
    validation = validate_preregistration_bundle(bundle)
    manifest = load_manifest(bundle / "manifest.json")
    environment = validate_execution_environment(manifest)
    source_validation = validate_source_evidence(repo, manifest)
    registry = json.loads((bundle / "analytic_registry.json").read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in registry["targets"]}
    results = [
        estimate_target(manifest, target, by_id[target.identifier])
        for target in manifest.targets
    ]
    for target, result in zip(manifest.targets, results, strict=True):
        validate_target_result(manifest, target, by_id[target.identifier], result)
    analysis = analyze_target_results(manifest, registry, results)
    payload = {
        "schema_version": 1,
        "experiment_id": "ACL-007",
        "kind": "confirmatory-cross-class-sequential-particle-filter-transport",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "source_evidence_validation": source_validation,
        "confirmatory_environment": environment,
        "randomness": "PCG64 independent target streams",
        "target_refit": False,
        "source_experiment": manifest.source_experiment,
        "source_rule": manifest.source_rule,
        "benchmark_scope": manifest.benchmark_scope,
        "inference_scope": manifest.inference_scope,
        "transport_scope": manifest.transport_scope,
        "frozen_design": manifest.raw,
        "locked_analytic_registry": registry,
        "preregistration_bundle_lock": json.loads(
            (bundle / "LOCK.json").read_text(encoding="utf-8")
        ),
        "target_results": results,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(canonical, payload)
