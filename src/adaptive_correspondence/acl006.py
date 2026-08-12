"""ACL-006 exact support-conditioned bias and self-consistency protocol."""

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
    type7_quantile,
    validate_lock,
)
from .io import provenance, write_json
from .support_bias import BlockSpecification, exact_block_moments, fisher_cosine

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ACL006_ACTIONS = 3
ACL006_FISHER_RCOND = 1e-12
ACL006_REPLICATION_SCHEDULE = (8192, 32768, 131072, 262144)
ACL006_CHUNK_SIZE = 4096
ACL006_DIRECTION_SCORE_MAX = 5.0
ACL006_FULL_SCORE_MEDIAN_MAX = 1.5
ACL006_FULL_SCORE_Q90_MAX = 2.5
ACL006_DISSOCIATION_EXACT_TRUTH_MAX = 0.90
ACL006_DISSOCIATION_OBSERVED_TRUTH_MAX = 0.95
ACL006_DISSOCIATION_HALF_COSINE_MIN = 0.995
ACL006_RESOLVABLE_CONTRAST_GAP_MIN = 0.10
ACL006_REGISTRY_ATOL = 2e-12
ACL006_REGISTRY_RTOL = 2e-12
ACL006_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.13.14",
    "numpy_version": "2.5.2",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
}
ACL006_TARGET_IDS = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "F07",
    "F08",
    "O01",
    "O02",
    "O03",
    "O04",
    "K01",
    "K02",
    "K03",
    "K04",
)
ACL006_CONTRAST_IDS = (
    "effective-count-N16",
    "effective-count-N32",
    "effective-count-N64",
    "effective-count-N128",
    "reward-shift-rare-context-N64",
    "reward-shift-rare-action-N64",
    "reward-shift-balanced-N32",
    "conditioning-centered-N32",
    "conditioning-rotated-N64",
)
ACL006_DISSOCIATION_IDS = ("F02", "F04", "F06", "O02", "K01", "K02", "K03")
ACL006_RESOLVABLE_CONTRAST_IDS = (
    "effective-count-N16",
    "effective-count-N32",
    "effective-count-N64",
    "reward-shift-rare-action-N64",
)
ACL006_LOCKED_FILES = frozenset(
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
class ACL006Target:
    identifier: str
    family: str
    sample_count: int
    specification: BlockSpecification
    seed: int


@dataclass(frozen=True)
class ACL006Contrast:
    identifier: str
    kind: str
    left: str
    right: str


@dataclass(frozen=True)
class ACL006Manifest:
    experiment_id: str
    actions: int
    empirical_fisher_rcond: float
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
    targets: tuple[ACL006Target, ...]
    contrasts: tuple[ACL006Contrast, ...]
    benchmark_scope: str
    inference_scope: str
    mechanism_scope: str
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


def validate_manifest_dict(payload: dict[str, Any]) -> ACL006Manifest:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ACL-006 manifest schema")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if payload.get("randomness") != "PCG64-independent-target-streams":
        raise ValueError("ACL-006 randomness mismatch")
    actions = _integer(payload, "actions", minimum=2)
    rcond = _positive_float(payload, "empirical_fisher_rcond")
    schedule_payload = payload.get("replication_schedule")
    if not isinstance(schedule_payload, list) or not schedule_payload:
        raise ValueError("replication_schedule must be a non-empty list")
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
    if contrast_gap > 2.0:
        raise ValueError("resolvable contrast gap cannot exceed two")
    registry_atol = _positive_float(payload, "analytic_registry_atol")
    registry_rtol = _positive_float(payload, "analytic_registry_rtol")

    target_payload = payload.get("targets")
    if not isinstance(target_payload, list) or not target_payload:
        raise ValueError("targets must be a non-empty list")
    targets: list[ACL006Target] = []
    identifiers: set[str] = set()
    seeds: set[int] = set()
    for item in target_payload:
        if not isinstance(item, dict):
            raise ValueError("each ACL-006 target must be an object")
        identifier = item.get("id")
        family = item.get("family")
        seed = item.get("seed")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("target IDs must be non-empty and unique")
        if not isinstance(family, str) or not family:
            raise ValueError("target family must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in seeds:
            raise ValueError("target seeds must be distinct non-negative integers")
        sample_count = _integer(item, "sample_count")
        try:
            context_probability = float(item.get("context_probability"))
        except (TypeError, ValueError) as error:
            raise ValueError("context_probability must be numeric") from error
        specification = BlockSpecification(
            policy=item.get("policy"),
            reward=item.get("reward"),
            context_probability=context_probability,
        )
        policy, _, _ = specification.arrays()
        if policy.size != actions:
            raise ValueError("target action count differs from manifest")
        targets.append(
            ACL006Target(identifier, family, sample_count, specification, seed)
        )
        identifiers.add(identifier)
        seeds.add(seed)

    contrast_payload = payload.get("contrasts")
    if not isinstance(contrast_payload, list) or not contrast_payload:
        raise ValueError("contrasts must be a non-empty list")
    contrasts: list[ACL006Contrast] = []
    contrast_ids: set[str] = set()
    for item in contrast_payload:
        if not isinstance(item, dict):
            raise ValueError("each ACL-006 contrast must be an object")
        identifier = item.get("id")
        kind = item.get("kind")
        left = item.get("left")
        right = item.get("right")
        if not isinstance(identifier, str) or not identifier or identifier in contrast_ids:
            raise ValueError("contrast IDs must be non-empty and unique")
        if not isinstance(kind, str) or not kind:
            raise ValueError("contrast kind must be non-empty")
        if left not in identifiers or right not in identifiers or left == right:
            raise ValueError("contrast endpoints must be distinct frozen targets")
        contrasts.append(ACL006Contrast(identifier, kind, left, right))
        contrast_ids.add(identifier)

    target_by_id = {target.identifier: target for target in targets}
    for contrast in contrasts:
        left = target_by_id[contrast.left]
        right = target_by_id[contrast.right]
        left_policy, left_reward, left_rho = left.specification.arrays()
        right_policy, right_reward, right_rho = right.specification.arrays()
        if contrast.kind == "matched-effective-count-support-factorization":
            left_effective = left.sample_count * left_rho * float(np.min(left_policy))
            right_effective = right.sample_count * right_rho * float(np.min(right_policy))
            if (
                left.sample_count != right.sample_count
                or not np.array_equal(left_reward, right_reward)
                or not np.isclose(left_effective, right_effective, atol=2e-14, rtol=0.0)
                or (left_rho == right_rho and np.array_equal(left_policy, right_policy))
            ):
                raise ValueError("matched-effective-count contrast semantics are invalid")
        elif contrast.kind == "reward-shift":
            shift = right_reward - left_reward
            if (
                left.sample_count != right.sample_count
                or left_rho != right_rho
                or not np.array_equal(left_policy, right_policy)
                or np.all(shift == 0.0)
                or not np.allclose(shift, shift[0], atol=2e-14, rtol=0.0)
            ):
                raise ValueError("reward-shift contrast semantics are invalid")
        elif contrast.kind == "fisher-conditioning":
            if (
                left.sample_count != right.sample_count
                or left_rho != right_rho
                or not np.array_equal(left_reward, right_reward)
                or not np.isclose(
                    np.min(left_policy), np.min(right_policy), atol=2e-14, rtol=0.0
                )
                or np.array_equal(left_policy, right_policy)
            ):
                raise ValueError("Fisher-conditioning contrast semantics are invalid")
        else:
            raise ValueError("unsupported ACL-006 contrast kind")

    scopes = tuple(
        payload.get(key)
        for key in ("benchmark_scope", "inference_scope", "mechanism_scope")
    )
    if not all(isinstance(value, str) and value for value in scopes):
        raise ValueError("ACL-006 scope fields must be non-empty")
    environment = payload.get("confirmatory_environment")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("confirmatory_environment must be a string mapping")

    if experiment_id == "ACL-006":
        exact = (
            actions == ACL006_ACTIONS
            and rcond == ACL006_FISHER_RCOND
            and schedule == ACL006_REPLICATION_SCHEDULE
            and chunk_size == ACL006_CHUNK_SIZE
            and direction_score_max == ACL006_DIRECTION_SCORE_MAX
            and median_max == ACL006_FULL_SCORE_MEDIAN_MAX
            and q90_max == ACL006_FULL_SCORE_Q90_MAX
            and dissociation_exact == ACL006_DISSOCIATION_EXACT_TRUTH_MAX
            and dissociation_observed == ACL006_DISSOCIATION_OBSERVED_TRUTH_MAX
            and dissociation_half == ACL006_DISSOCIATION_HALF_COSINE_MIN
            and contrast_gap == ACL006_RESOLVABLE_CONTRAST_GAP_MIN
            and registry_atol == ACL006_REGISTRY_ATOL
            and registry_rtol == ACL006_REGISTRY_RTOL
            and tuple(target.identifier for target in targets) == ACL006_TARGET_IDS
            and tuple(contrast.identifier for contrast in contrasts)
            == ACL006_CONTRAST_IDS
            and environment == ACL006_ENVIRONMENT
            and scopes
            == (
                "deterministic-held-out-support-bias-benchmark",
                "descriptive-criteria-not-population-confidence",
                "undamped-three-action-empirical-fisher-plugin-family",
            )
        )
        if not exact:
            raise ValueError("ACL-006 design constants mismatch")

    return ACL006Manifest(
        experiment_id=experiment_id,
        actions=actions,
        empirical_fisher_rcond=rcond,
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
        targets=tuple(targets),
        contrasts=tuple(contrasts),
        benchmark_scope=scopes[0],
        inference_scope=scopes[1],
        mechanism_scope=scopes[2],
        confirmatory_environment=dict(environment),
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL006Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-006 manifest") from error
    return validate_manifest_dict(payload)


def _fisher(specification: BlockSpecification) -> FloatArray:
    policy, _, rho = specification.arrays()
    return rho * (np.diag(policy) - np.outer(policy, policy))


def _fisher_norm(metric: FloatArray, direction: ArrayLike) -> float:
    vector = np.asarray(direction, dtype=np.float64)
    squared = float(vector @ metric @ vector)
    if not np.isfinite(squared) or squared < -2e-13:
        raise FloatingPointError("invalid Fisher norm")
    return float(np.sqrt(max(squared, 0.0)))


def _angular_envelope(
    *, mean_norm: float, rms_standard_error: float, score_max: float
) -> float:
    radius = score_max * rms_standard_error
    if radius >= mean_norm:
        return 2.0
    return float(min(2.0, 2.0 * radius / (mean_norm - radius)))


def _half_cosine_lower_bound(
    *, mean_norm: float, half_rms_standard_error: float, score_max: float
) -> float:
    radius = score_max * half_rms_standard_error
    if radius >= mean_norm:
        return -1.0
    unit_error_bound = 2.0 * radius / (mean_norm - radius)
    return float(max(-1.0, 1.0 - 2.0 * unit_error_bound**2))


def _support_key(support: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in support)


def build_analytic_registry(manifest: ACL006Manifest) -> dict[str, Any]:
    final_replications = manifest.replication_schedule[-1]
    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for target in manifest.targets:
        moments = exact_block_moments(
            target.specification,
            sample_count=target.sample_count,
            rcond=manifest.empirical_fisher_rcond,
        )
        policy, reward, rho = target.specification.arrays()
        metric = _fisher(target.specification)
        eigenvalues = np.linalg.eigvalsh(metric)
        positive = eigenvalues[eigenvalues > 2e-14]
        if positive.size != 2:
            raise FloatingPointError("ACL-006 analytic Fisher must have rank two")
        mean_norm = _fisher_norm(metric, moments.mean_direction)
        if mean_norm <= 1e-12:
            raise FloatingPointError("ACL-006 exact estimator mean is too small")
        variance_trace = float(np.trace(metric @ moments.covariance))
        if not np.isfinite(variance_trace) or variance_trace <= 0.0:
            raise FloatingPointError("ACL-006 exact Fisher variance must be positive")
        full_se = float(np.sqrt(variance_trace / final_replications))
        half_se = float(np.sqrt(variance_trace / (final_replications / 2)))
        envelope = _angular_envelope(
            mean_norm=mean_norm,
            rms_standard_error=full_se,
            score_max=manifest.direction_score_max,
        )
        half_lower = _half_cosine_lower_bound(
            mean_norm=mean_norm,
            half_rms_standard_error=half_se,
            score_max=manifest.direction_score_max,
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
            "seed": target.seed,
            "sample_count": target.sample_count,
            "context_probability": rho,
            "policy": policy.tolist(),
            "reward": reward.tolist(),
            "joint_action_probabilities": (rho * policy).tolist(),
            "minimum_joint_cell_probability": float(rho * np.min(policy)),
            "effective_minimum_count": float(target.sample_count * rho * np.min(policy)),
            "analytic_fisher": metric.tolist(),
            "positive_fisher_eigenvalues": positive.tolist(),
            "positive_fisher_condition_number": float(positive[-1] / positive[0]),
            "analytic_direction": moments.analytic_direction.tolist(),
            "exact_mean_direction": moments.mean_direction.tolist(),
            "exact_direction_covariance": moments.covariance.tolist(),
            "exact_probability_mass": moments.probability_mass,
            "exact_truth_alignment_cosine": exact_cosine,
            "exact_truth_angular_loss": 1.0 - exact_cosine,
            "rank_deficient_probability": moments.rank_deficient_probability,
            "support_probabilities": {
                _support_key(key): value
                for key, value in sorted(moments.support_probabilities.items())
            },
            "mean_support_loss": moments.mean_support_loss.tolist(),
            "mean_observed_support_perturbation": (
                moments.mean_observed_support_perturbation.tolist()
            ),
            "expected_squared_fisher_error": moments.expected_squared_error,
            "expected_squared_support_loss": moments.expected_squared_support_loss,
            "expected_squared_observed_support_perturbation": (
                moments.expected_squared_observed_support_perturbation
            ),
            "exact_mean_fisher_norm": mean_norm,
            "single_shadow_fisher_variance_trace": variance_trace,
            "final_replications": final_replications,
            "final_fisher_rms_standard_error": full_se,
            "half_fisher_rms_standard_error": half_se,
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
        conservative_gap = (
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
                "conservative_absolute_gap": conservative_gap,
                "resolvable": (
                    conservative_gap >= manifest.resolvable_contrast_gap_min
                ),
            }
        )

    dissociation_ids = [entry["id"] for entry in entries if entry["dissociation_stratum"]]
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "exact-finite-support-conditioned-bias-registry",
        "outcomes_generated": False,
        "shadow_count": 0,
        "target_refit": False,
        "comparator_source": "exact-four-cell-multinomial-enumeration",
        "numeric_comparison": {
            "absolute_tolerance": manifest.analytic_registry_atol,
            "relative_tolerance": manifest.analytic_registry_rtol,
        },
        "targets": entries,
        "contrasts": contrast_entries,
        "dissociation_target_ids": dissociation_ids,
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


def direct_plugin_direction(
    *,
    policy: ArrayLike,
    reward: ArrayLike,
    counts: ArrayLike,
    sample_count: int,
    rcond: float,
) -> FloatArray:
    """Compute one realized direction in full coordinates, independent of enumeration."""
    specification = BlockSpecification(policy=policy, reward=reward, context_probability=0.5)
    policy_array, reward_array, _ = specification.arrays()
    count_array = np.asarray(counts)
    if count_array.shape != (4,) or not np.issubdtype(count_array.dtype, np.integer):
        raise ValueError("counts must contain four integers")
    if np.any(count_array < 0) or int(np.sum(count_array)) != sample_count:
        raise ValueError("counts must be non-negative and sum to sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    scores = np.eye(3, dtype=np.float64) - policy_array[None, :]
    action_counts = count_array[:3].astype(np.float64)
    fisher = np.einsum("a,ai,aj->ij", action_counts, scores, scores) / sample_count
    gradient = (action_counts * reward_array) @ scores / sample_count
    direction = np.linalg.pinv(fisher, rcond=rcond, hermitian=True) @ gradient
    direction -= np.mean(direction)
    if not np.all(np.isfinite(direction)):
        raise FloatingPointError("direct plug-in direction is non-finite")
    return direction.astype(np.float64, copy=False)


def batch_plugin_directions_from_counts(
    *,
    policy: ArrayLike,
    reward: ArrayLike,
    counts: ArrayLike,
    sample_count: int,
    rcond: float,
) -> tuple[FloatArray, IntArray]:
    """Vectorized full-coordinate path used by stochastic ACL-006 execution."""
    specification = BlockSpecification(policy=policy, reward=reward, context_probability=0.5)
    policy_array, reward_array, _ = specification.arrays()
    count_array = np.asarray(counts)
    if (
        count_array.ndim != 2
        or count_array.shape[1] != 4
        or not np.issubdtype(count_array.dtype, np.integer)
    ):
        raise ValueError("batch counts must have shape (replications, 4) and integer dtype")
    if np.any(count_array < 0) or np.any(np.sum(count_array, axis=1) != sample_count):
        raise ValueError("each batch count row must be non-negative and sum to sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    action_counts = count_array[:, :3].astype(np.float64)
    scores = np.eye(3, dtype=np.float64) - policy_array[None, :]
    score_outer = scores[:, :, None] * scores[:, None, :]
    fishers = np.einsum("ba,aij->bij", action_counts, score_outer) / sample_count
    gradients = (action_counts * reward_array[None, :]) @ scores / sample_count
    inverses = np.linalg.pinv(fishers, rcond=rcond, hermitian=True)
    directions = np.einsum("bij,bj->bi", inverses, gradients)
    directions -= np.mean(directions, axis=1, keepdims=True)
    if not np.all(np.isfinite(directions)):
        raise FloatingPointError("batched plug-in direction is non-finite")
    masks = np.sum(
        (count_array[:, :3] > 0).astype(np.int64)
        * (1 << np.arange(3, dtype=np.int64))[None, :],
        axis=1,
    )
    return directions.astype(np.float64, copy=False), masks


def _sample_plugin_directions(
    target: ACL006Target,
    *,
    replications: int,
    rng: np.random.Generator,
    rcond: float,
) -> tuple[FloatArray, IntArray]:
    policy, reward, rho = target.specification.arrays()
    probabilities = np.concatenate((rho * policy, [1.0 - rho]))
    counts = rng.multinomial(target.sample_count, probabilities, size=replications)
    return batch_plugin_directions_from_counts(
        policy=policy,
        reward=reward,
        counts=counts,
        sample_count=target.sample_count,
        rcond=rcond,
    )


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
    metric: FloatArray,
    observed: FloatArray,
    exact: FloatArray,
    rms_standard_error: float,
) -> float:
    if rms_standard_error <= 0.0 or not np.isfinite(rms_standard_error):
        raise FloatingPointError("direction score scale must be finite and positive")
    return _fisher_norm(metric, observed - exact) / rms_standard_error


def estimate_target(
    manifest: ACL006Manifest,
    target: ACL006Target,
    registry_entry: dict[str, Any],
) -> dict[str, Any]:
    exact_mean = np.asarray(registry_entry["exact_mean_direction"], dtype=np.float64)
    analytic = np.asarray(registry_entry["analytic_direction"], dtype=np.float64)
    metric = np.asarray(registry_entry["analytic_fisher"], dtype=np.float64)
    variance_trace = float(registry_entry["single_shadow_fisher_variance_trace"])
    rng = np.random.Generator(np.random.PCG64(target.seed))
    chunks: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints = set(manifest.replication_schedule)
    generated = 0
    while generated < manifest.replication_schedule[-1]:
        directions, masks = _sample_plugin_directions(
            target,
            replications=manifest.chunk_size,
            rng=rng,
            rcond=manifest.empirical_fisher_rcond,
        )
        chunks.append(
            {
                "chunk_index": len(chunks),
                "count": manifest.chunk_size,
                "direction_sum": np.sum(directions, axis=0).tolist(),
                "direction_outer_sum": (directions.T @ directions).tolist(),
                "support_mask_counts": np.bincount(masks, minlength=8).tolist(),
            }
        )
        generated += manifest.chunk_size
        if generated not in checkpoints:
            continue
        half_chunks = len(chunks) // 2
        full_mean = _mean_from_chunks(chunks)
        first_mean = _mean_from_chunks(chunks[:half_chunks])
        second_mean = _mean_from_chunks(chunks[half_chunks:])
        full_se = float(np.sqrt(variance_trace / generated))
        half_se = float(np.sqrt(variance_trace / (generated / 2)))
        history.append(
            {
                "replications": generated,
                "full_mean_direction": full_mean.tolist(),
                "first_half_mean_direction": first_mean.tolist(),
                "second_half_mean_direction": second_mean.tolist(),
                "full_direction_score": _direction_score(
                    metric, full_mean, exact_mean, full_se
                ),
                "first_half_direction_score": _direction_score(
                    metric, first_mean, exact_mean, half_se
                ),
                "second_half_direction_score": _direction_score(
                    metric, second_mean, exact_mean, half_se
                ),
                "full_truth_cosine": fisher_cosine(
                    target.specification, full_mean, analytic
                ),
                "half_cosine": fisher_cosine(
                    target.specification, first_mean, second_mean
                ),
            }
        )
    final = history[-1]
    observed_truth = float(final["full_truth_cosine"])
    exact_truth = float(registry_entry["exact_truth_alignment_cosine"])
    return {
        "target_id": target.identifier,
        "family": target.family,
        "seed": target.seed,
        "sample_count": target.sample_count,
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
    manifest: ACL006Manifest,
    target: ACL006Target,
    registry_entry: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Recompute every stored checkpoint from sufficient statistics."""
    if result.get("target_id") != target.identifier or result.get("seed") != target.seed:
        raise ValueError("ACL-006 target result identity differs from manifest")
    chunks = result.get("shadow_chunks")
    history = result.get("checkpoint_history")
    if not isinstance(chunks, list) or not isinstance(history, list):
        raise ValueError("ACL-006 target result lacks chunk/checkpoint data")
    expected_chunk_count = manifest.replication_schedule[-1] // manifest.chunk_size
    if len(chunks) != expected_chunk_count or len(history) != len(
        manifest.replication_schedule
    ):
        raise ValueError("ACL-006 target result has the wrong fixed budget")
    for index, chunk in enumerate(chunks):
        if chunk.get("chunk_index") != index or chunk.get("count") != manifest.chunk_size:
            raise ValueError("ACL-006 target chunk metadata is invalid")
        support_counts = np.asarray(chunk.get("support_mask_counts"), dtype=np.int64)
        outer = np.asarray(chunk.get("direction_outer_sum"), dtype=np.float64)
        if (
            support_counts.shape != (8,)
            or int(np.sum(support_counts)) != manifest.chunk_size
            or np.any(support_counts < 0)
            or outer.shape != (3, 3)
            or not np.all(np.isfinite(outer))
        ):
            raise ValueError("ACL-006 target chunk sufficient statistics are invalid")

    exact_mean = np.asarray(registry_entry["exact_mean_direction"], dtype=np.float64)
    analytic = np.asarray(registry_entry["analytic_direction"], dtype=np.float64)
    metric = np.asarray(registry_entry["analytic_fisher"], dtype=np.float64)
    variance_trace = float(registry_entry["single_shadow_fisher_variance_trace"])
    for checkpoint, stored in zip(
        manifest.replication_schedule, history, strict=True
    ):
        prefix = chunks[: checkpoint // manifest.chunk_size]
        midpoint = len(prefix) // 2
        full_mean = _mean_from_chunks(prefix)
        first_mean = _mean_from_chunks(prefix[:midpoint])
        second_mean = _mean_from_chunks(prefix[midpoint:])
        full_se = float(np.sqrt(variance_trace / checkpoint))
        half_se = float(np.sqrt(variance_trace / (checkpoint / 2)))
        expected_scalars = {
            "full_direction_score": _direction_score(
                metric, full_mean, exact_mean, full_se
            ),
            "first_half_direction_score": _direction_score(
                metric, first_mean, exact_mean, half_se
            ),
            "second_half_direction_score": _direction_score(
                metric, second_mean, exact_mean, half_se
            ),
            "full_truth_cosine": fisher_cosine(
                target.specification, full_mean, analytic
            ),
            "half_cosine": fisher_cosine(
                target.specification, first_mean, second_mean
            ),
        }
        if stored.get("replications") != checkpoint:
            raise ValueError("ACL-006 checkpoint schedule differs from manifest")
        for key, value in (
            ("full_mean_direction", full_mean),
            ("first_half_mean_direction", first_mean),
            ("second_half_mean_direction", second_mean),
        ):
            if not np.array_equal(np.asarray(stored.get(key), dtype=np.float64), value):
                raise FloatingPointError(f"ACL-006 chunks do not reproduce {key}")
        for key, value in expected_scalars.items():
            if not np.isclose(
                float(stored.get(key)), value, atol=5e-14, rtol=5e-14
            ):
                raise FloatingPointError(f"ACL-006 chunks do not reproduce {key}")
    final = history[-1]
    if not np.array_equal(
        np.asarray(result.get("observed_mean_direction"), dtype=np.float64),
        np.asarray(final["full_mean_direction"], dtype=np.float64),
    ):
        raise FloatingPointError("ACL-006 final result does not match final checkpoint")
    final_fields = {
        "full_direction_score": float(final["full_direction_score"]),
        "first_half_direction_score": float(final["first_half_direction_score"]),
        "second_half_direction_score": float(final["second_half_direction_score"]),
        "observed_truth_cosine": float(final["full_truth_cosine"]),
        "final_half_cosine": float(final["half_cosine"]),
    }
    for key, expected in final_fields.items():
        actual = float(result.get(key))
        if not np.isfinite(actual) or actual != expected:
            raise FloatingPointError(f"ACL-006 final result does not match {key}")
    exact_truth = float(registry_entry["exact_truth_alignment_cosine"])
    if float(result.get("exact_truth_cosine")) != exact_truth or not np.isclose(
        float(result.get("angular_residual")),
        abs(final_fields["observed_truth_cosine"] - exact_truth),
        atol=5e-16,
        rtol=5e-16,
    ):
        raise FloatingPointError("ACL-006 final angular residual is invalid")
    if (
        float(result.get("angular_envelope"))
        != float(registry_entry["final_angular_envelope"])
        or bool(result.get("dissociation_stratum"))
        is not bool(registry_entry["dissociation_stratum"])
    ):
        raise ValueError("ACL-006 final analytic labels differ from registry")


def analyze_target_results(
    manifest: ACL006Manifest,
    registry: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_ids = [target.identifier for target in manifest.targets]
    if [row.get("target_id") for row in results] != expected_ids:
        raise ValueError("ACL-006 result target IDs or order differ from manifest")
    registry_by_id = {entry["id"]: entry for entry in registry["targets"]}
    rows_by_id = {row["target_id"]: row for row in results}
    numeric_keys = (
        "full_direction_score",
        "first_half_direction_score",
        "second_half_direction_score",
        "angular_residual",
        "angular_envelope",
        "observed_truth_cosine",
        "final_half_cosine",
    )
    if any(
        not np.isfinite(float(row[key])) for row in results for key in numeric_keys
    ):
        raise FloatingPointError("ACL-006 analysis received a non-finite result")
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
    primary_pass = (
        max(all_scores) <= manifest.direction_score_max
        and median <= manifest.full_score_median_max
        and q90 <= manifest.full_score_q90_max
        and angular_ok
    )

    dissociation_rows = [
        row
        for row in results
        if registry_by_id[row["target_id"]]["dissociation_stratum"]
    ]
    if not dissociation_rows:
        dissociation_verdict = "NOT_APPLICABLE"
    elif all(
        float(row["final_half_cosine"])
        >= manifest.dissociation_half_cosine_min
        and float(row["observed_truth_cosine"])
        <= manifest.dissociation_observed_truth_cosine_max
        for row in dissociation_rows
    ):
        dissociation_verdict = "PASS"
    else:
        dissociation_verdict = "FAIL"

    contrast_rows = []
    resolvable_passes = []
    for contrast in registry["contrasts"]:
        left = rows_by_id[contrast["left"]]
        right = rows_by_id[contrast["right"]]
        observed_gap = (
            float(right["observed_truth_cosine"])
            - float(left["observed_truth_cosine"])
        )
        predicted_gap = float(contrast["exact_signed_truth_cosine_gap"])
        sign_agrees = observed_gap * predicted_gap > 0.0
        retains_gap = abs(observed_gap) >= manifest.resolvable_contrast_gap_min
        reproduced = sign_agrees and retains_gap
        if contrast["resolvable"]:
            resolvable_passes.append(reproduced)
        contrast_rows.append(
            {
                **contrast,
                "observed_signed_truth_cosine_gap": observed_gap,
                "predicted_sign_reproduced": sign_agrees,
                "observed_gap_retains_minimum": retains_gap,
                "gating_reproduced": reproduced if contrast["resolvable"] else None,
            }
        )
    if not resolvable_passes:
        contrast_verdict = "NOT_APPLICABLE"
    else:
        contrast_verdict = "PASS" if all(resolvable_passes) else "FAIL"

    effective_counterexample = any(
        row["resolvable"]
        and row["kind"] == "matched-effective-count-support-factorization"
        for row in registry["contrasts"]
    )
    reward_counterexample = any(
        row["resolvable"] and row["kind"] == "reward-shift"
        for row in registry["contrasts"]
    )
    return {
        "experiment_id": manifest.experiment_id,
        "exact_mean_prediction_verdict": "PASS" if primary_pass else "FAIL",
        "dissociation_prediction_verdict": dissociation_verdict,
        "stochastic_contrast_reproduction_verdict": contrast_verdict,
        "direction_score_maximum": max(all_scores),
        "full_direction_score_median_type7": median,
        "full_direction_score_q90_type7": q90,
        "all_angular_residuals_within_frozen_envelopes": angular_ok,
        "dissociation_target_ids": [row["target_id"] for row in dissociation_rows],
        "self_consistency_certifies_truth": False,
        "effective_minimum_count_only_law": (
            "FALSIFIED_BY_EXACT_COUNTEREXAMPLE"
            if effective_counterexample
            else "UNRESOLVED"
        ),
        "support_and_fisher_spectrum_only_law": (
            "FALSIFIED_BY_EXACT_REWARD_SHIFT_COUNTEREXAMPLE"
            if reward_counterexample
            else "UNRESOLVED"
        ),
        "target_refit": False,
        "fixed_replication_budget": manifest.replication_schedule[-1],
        "population_confidence_claim": False,
        "contrast_results": contrast_rows,
    }


def validate_execution_environment(manifest: ACL006Manifest) -> dict[str, Any]:
    actual = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    if actual != manifest.confirmatory_environment:
        raise ValueError("ACL-006 confirmatory environment mismatch")
    return {"valid": True, **actual, "dtype": "float64", "rng": "PCG64"}


def validate_preregistration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if (
        lock.get("experiment_id") != "ACL-006"
        or lock.get("kind") != "preregistration-bundle-lock"
        or lock.get("outcomes_generated") is not False
        or set(lock.get("files", {})) != ACL006_LOCKED_FILES
    ):
        raise ValueError("ACL-006 lock must contain the exact frozen file set")
    actual_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if actual_files != ACL006_LOCKED_FILES | {"LOCK.json"}:
        raise ValueError("ACL-006 bundle must have exact frozen directory contents")
    manifest = load_manifest(bundle / "manifest.json")
    try:
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-006 analytic registry") from error
    recomputed = build_analytic_registry(manifest)
    _assert_numeric_equivalence(
        locked_registry,
        recomputed,
        atol=manifest.analytic_registry_atol,
        rtol=manifest.analytic_registry_rtol,
    )
    if locked_registry.get("outcomes_generated") is not False:
        raise ValueError("ACL-006 registry must remain analytic-only")
    if manifest.experiment_id == "ACL-006" and (
        tuple(locked_registry["dissociation_target_ids"])
        != ACL006_DISSOCIATION_IDS
        or tuple(locked_registry["resolvable_contrast_ids"])
        != ACL006_RESOLVABLE_CONTRAST_IDS
    ):
        raise ValueError("ACL-006 analytic strata differ from the frozen design")
    return {
        "schema_version": 1,
        "experiment_id": "ACL-006",
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "target_count": len(manifest.targets),
        "dissociation_target_count": len(locked_registry["dissociation_target_ids"]),
        "resolvable_contrast_count": len(locked_registry["resolvable_contrast_ids"]),
        "registry_comparison": "numeric-tolerance",
        "registry_absolute_tolerance": manifest.analytic_registry_atol,
        "registry_relative_tolerance": manifest.analytic_registry_rtol,
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
    canonical = (repo / "evidence" / f"ACL-006-confirmatory-{approved_sha}.json").resolve()
    if requested != canonical:
        raise ValueError("ACL-006 output must equal the SHA-derived canonical evidence path")
    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = repo / bundle
    bundle = bundle.resolve()
    canonical_bundle = (repo / "preregistrations" / "ACL-006").resolve()
    if bundle != canonical_bundle:
        raise ValueError("ACL-006 requires the SHA-bound canonical preregistration bundle")
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
    locked_registry = json.loads(
        (bundle / "analytic_registry.json").read_text(encoding="utf-8")
    )
    registry_by_id = {entry["id"]: entry for entry in locked_registry["targets"]}
    results = [
        estimate_target(manifest, target, registry_by_id[target.identifier])
        for target in manifest.targets
    ]
    for target, result in zip(manifest.targets, results, strict=True):
        validate_target_result(
            manifest, target, registry_by_id[target.identifier], result
        )
    analysis = analyze_target_results(manifest, locked_registry, results)
    payload = {
        "schema_version": 1,
        "experiment_id": "ACL-006",
        "kind": "confirmatory-exact-support-conditioned-angular-bias",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "confirmatory_environment": environment,
        "randomness": "PCG64 independent target streams",
        "target_refit": False,
        "benchmark_scope": manifest.benchmark_scope,
        "inference_scope": manifest.inference_scope,
        "mechanism_scope": manifest.mechanism_scope,
        "frozen_design": manifest.raw,
        "locked_analytic_registry": locked_registry,
        "preregistration_bundle_lock": json.loads(
            (bundle / "LOCK.json").read_text(encoding="utf-8")
        ),
        "target_results": results,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(canonical, payload)
