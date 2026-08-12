"""ACL-004 finite-lambda Gaussian conditional-mean protocol."""

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
    type7_quantile,
    validate_lock,
)
from .gaussian_rank_mu_bridge import (
    GaussianLinearBridgeState,
    analytic_finite_lambda_direction,
    fisher_block_cosines,
    sample_rank_mu_shadows,
)
from .io import provenance, write_json

FloatArray = NDArray[np.float64]

ACL004_DIMENSION = 3
ACL004_SAMPLE_COUNT = 32
ACL004_PARENT_COUNT = 16
ACL004_MEAN_LEARNING_RATE = 0.2
ACL004_COVARIANCE_LEARNING_RATE = 0.1
ACL004_QUADRATURE_ORDER = 160
ACL004_QUADRATURE_ORACLE_ORDER = 320
ACL004_QUADRATURE_RTOL = 2e-9
ACL004_QUADRATURE_ATOL = 5e-12
ACL004_REPLICATION_SCHEDULE = (4096, 8192, 16384, 32768, 65536)
ACL004_CHUNK_SIZE = 2048
ACL004_HALF_COSINE_MIN = 0.98
ACL004_H2_COSINE_MIN = 0.99
ACL004_H1_SHADOW_COUNT = 2048
ACL004_LOCKED_FILES = frozenset(
    {
        "ANALYSIS_PLAN.md",
        "DERIVATION.md",
        "PREREGISTRATION.md",
        "README.md",
        "analytic_registry.json",
        "manifest.json",
    }
)
RANK_WEIGHT_DESCRIPTION = "log(parent_count+0.5)-log(rank), normalized"


@dataclass(frozen=True)
class ACL004Landscape:
    identifier: str
    state: GaussianLinearBridgeState
    seed: int


@dataclass(frozen=True)
class ACL004Manifest:
    experiment_id: str
    dimension: int
    sample_count: int
    parent_count: int
    mean_learning_rate: float
    covariance_learning_rate: float
    quadrature_order: int
    quadrature_oracle_order: int
    quadrature_rtol: float
    quadrature_atol: float
    replication_schedule: tuple[int, ...]
    chunk_size: int
    half_cosine_min: float
    h2_cosine_min: float
    h1_shadow_count: int
    landscapes: tuple[ACL004Landscape, ...]
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


def validate_manifest_dict(payload: dict[str, Any]) -> ACL004Manifest:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ACL-004 manifest schema")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if (
        payload.get("randomness") != "PCG64-independent-landscape-streams"
        or payload.get("parameterization") != "mean-and-log-standard-deviation"
        or payload.get("rank_weights") != RANK_WEIGHT_DESCRIPTION
    ):
        raise ValueError("ACL-004 randomness, parameterization, or weights mismatch")
    dimension = _integer(payload, "dimension", minimum=2)
    sample_count = _integer(payload, "sample_count", minimum=2)
    parent_count = _integer(payload, "parent_count")
    if parent_count > sample_count:
        raise ValueError("parent_count cannot exceed sample_count")
    mean_learning_rate = _positive_float(payload, "mean_learning_rate")
    covariance_learning_rate = _positive_float(payload, "covariance_learning_rate")
    quadrature_order = _integer(payload, "quadrature_order", minimum=32)
    quadrature_oracle_order = _integer(payload, "quadrature_oracle_order", minimum=32)
    if quadrature_order > 320 or quadrature_oracle_order > 320:
        raise ValueError("quadrature orders exceed supported numerical range")
    quadrature_rtol = _positive_float(payload, "quadrature_relative_tolerance")
    quadrature_atol = _positive_float(payload, "quadrature_absolute_tolerance")
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

    landscape_payload = payload.get("landscapes")
    if not isinstance(landscape_payload, list) or not landscape_payload:
        raise ValueError("landscapes must be a non-empty list")
    landscapes = []
    identifiers: set[str] = set()
    seeds: set[int] = set()
    for item in landscape_payload:
        if not isinstance(item, dict):
            raise ValueError("each ACL-004 landscape must be an object")
        identifier = item.get("id")
        seed = item.get("seed")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("landscape IDs must be non-empty and unique")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in seeds:
            raise ValueError("landscape seeds must be distinct non-negative integers")
        state = GaussianLinearBridgeState(
            mean=item.get("mean"),
            log_std=item.get("log_std"),
            objective=item.get("objective"),
        )
        mean, _, _ = state.arrays()
        if mean.size != dimension:
            raise ValueError("landscape dimension differs from manifest")
        identifiers.add(identifier)
        seeds.add(seed)
        landscapes.append(ACL004Landscape(identifier, state, seed))
    scopes = tuple(
        payload.get(key) for key in ("benchmark_scope", "inference_scope", "transport_scope")
    )
    if not all(isinstance(value, str) and value for value in scopes):
        raise ValueError("ACL-004 scope fields must be non-empty")

    if experiment_id == "ACL-004":
        exact = (
            dimension == ACL004_DIMENSION
            and sample_count == ACL004_SAMPLE_COUNT
            and parent_count == ACL004_PARENT_COUNT
            and mean_learning_rate == ACL004_MEAN_LEARNING_RATE
            and covariance_learning_rate == ACL004_COVARIANCE_LEARNING_RATE
            and quadrature_order == ACL004_QUADRATURE_ORDER
            and quadrature_oracle_order == ACL004_QUADRATURE_ORACLE_ORDER
            and quadrature_rtol == ACL004_QUADRATURE_RTOL
            and quadrature_atol == ACL004_QUADRATURE_ATOL
            and schedule == ACL004_REPLICATION_SCHEDULE
            and chunk_size == ACL004_CHUNK_SIZE
            and half_cosine_min == ACL004_HALF_COSINE_MIN
            and h2_cosine_min == ACL004_H2_COSINE_MIN
            and h1_shadow_count == ACL004_H1_SHADOW_COUNT
            and len(landscapes) == 12
            and tuple(landscape.identifier for landscape in landscapes)
            == tuple(f"G{index:02d}" for index in range(1, 13))
            and scopes
            == (
                "deterministic-held-out-gaussian-linear-benchmark",
                "descriptive-criteria-not-population-confidence",
                "within-gaussian-class-no-target-refit",
            )
        )
        if not exact:
            raise ValueError("ACL-004 design constants mismatch")
    return ACL004Manifest(
        experiment_id=experiment_id,
        dimension=dimension,
        sample_count=sample_count,
        parent_count=parent_count,
        mean_learning_rate=mean_learning_rate,
        covariance_learning_rate=covariance_learning_rate,
        quadrature_order=quadrature_order,
        quadrature_oracle_order=quadrature_oracle_order,
        quadrature_rtol=quadrature_rtol,
        quadrature_atol=quadrature_atol,
        replication_schedule=schedule,
        chunk_size=chunk_size,
        half_cosine_min=half_cosine_min,
        h2_cosine_min=h2_cosine_min,
        h1_shadow_count=h1_shadow_count,
        landscapes=tuple(landscapes),
        benchmark_scope=scopes[0],
        inference_scope=scopes[1],
        transport_scope=scopes[2],
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL004Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-004 manifest") from error
    return validate_manifest_dict(payload)


def _fisher_norms(state: GaussianLinearBridgeState, direction: FloatArray) -> tuple[float, float]:
    mean, log_std, _ = state.arrays()
    dimension = mean.size
    mean_norm = float(
        np.sqrt(np.sum(np.exp(-2.0 * log_std) * direction[:dimension] ** 2))
    )
    covariance_norm = float(np.sqrt(2.0 * np.sum(direction[dimension:] ** 2)))
    return mean_norm, covariance_norm


def build_analytic_registry(manifest: ACL004Manifest) -> dict[str, Any]:
    entries = []
    for landscape in manifest.landscapes:
        analytic = analytic_finite_lambda_direction(
            landscape.state,
            sample_count=manifest.sample_count,
            parent_count=manifest.parent_count,
            quadrature_order=manifest.quadrature_order,
        )
        oracle = analytic_finite_lambda_direction(
            landscape.state,
            sample_count=manifest.sample_count,
            parent_count=manifest.parent_count,
            quadrature_order=manifest.quadrature_oracle_order,
        )
        if not np.allclose(
            analytic,
            oracle,
            rtol=manifest.quadrature_rtol,
            atol=manifest.quadrature_atol,
        ):
            raise FloatingPointError("ACL-004 analytic quadrature oracle mismatch")
        mean_norm, covariance_norm = _fisher_norms(landscape.state, analytic)
        if mean_norm <= 1e-12 or covariance_norm <= 1e-12:
            raise ValueError("ACL-004 analytic block is too small for directional testing")
        entries.append(
            {
                "id": landscape.identifier,
                "seed": landscape.seed,
                "analytic_direction": analytic.tolist(),
                "mean_block_fisher_norm": mean_norm,
                "covariance_block_fisher_norm": covariance_norm,
                "quadrature_oracle_max_absolute_error": float(
                    np.max(np.abs(analytic - oracle))
                ),
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "analytic-finite-lambda-gaussian-registry",
        "outcomes_generated": False,
        "shadow_count": 0,
        "comparator_source": "conditional-rank-utility-gaussian-score-inverse-fisher",
        "landscapes": entries,
    }


def _joint_fisher_cosine(
    state: GaussianLinearBridgeState, left: FloatArray, right: FloatArray
) -> float:
    mean, log_std, _ = state.arrays()
    metric = np.concatenate((np.exp(-2.0 * log_std), np.full(mean.size, 2.0)))
    numerator = float(np.sum(metric * left * right))
    denominator = float(
        np.sqrt(np.sum(metric * left**2) * np.sum(metric * right**2))
    )
    if denominator == 0.0:
        raise ValueError("joint Fisher cosine requires nonzero directions")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _many_block_cosines(
    state: GaussianLinearBridgeState, directions: FloatArray, analytic: FloatArray
) -> tuple[list[float], list[float]]:
    mean, log_std, _ = state.arrays()
    dimension = mean.size
    metrics = (np.exp(-2.0 * log_std), np.full(dimension, 2.0))
    output = []
    for block, metric in ((slice(0, dimension), metrics[0]), (slice(dimension, None), metrics[1])):
        values = directions[:, block]
        target = analytic[block]
        numerators = np.sum(metric[None, :] * values * target[None, :], axis=1)
        denominators = np.sqrt(
            np.sum(metric[None, :] * values**2, axis=1) * np.sum(metric * target**2)
        )
        cosines = np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators > 0.0,
        )
        if not np.all(np.isfinite(cosines)):
            raise FloatingPointError("single-shadow Fisher cosine is undefined")
        output.append(np.clip(cosines, -1.0, 1.0).tolist())
    return output[0], output[1]


def reproduce_stopped_mean(result: dict[str, Any]) -> FloatArray:
    chunks = result.get("shadow_chunks", [])
    if not chunks:
        raise ValueError("landscape result has no shadow chunks")
    total_count = sum(int(chunk["count"]) for chunk in chunks)
    total = np.sum(
        np.asarray([chunk["direction_sum"] for chunk in chunks], dtype=np.float64), axis=0
    )
    return total / total_count


def _mean_from_chunks(chunks: list[dict[str, Any]]) -> FloatArray:
    return reproduce_stopped_mean({"shadow_chunks": chunks})


def _uncertainty_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(chunk["count"]) for chunk in chunks)
    if count < 2:
        raise ValueError("shadow uncertainty requires at least two observations")
    mean = _mean_from_chunks(chunks)
    outer = np.sum(
        np.asarray([chunk["direction_outer_sum"] for chunk in chunks], dtype=np.float64),
        axis=0,
    )
    covariance = (outer - count * np.outer(mean, mean)) / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-14) or not np.all(np.isfinite(covariance)):
        raise FloatingPointError("invalid shadow covariance from sufficient statistics")
    standard_error = np.sqrt(np.maximum(diagonal, 0.0) / count)
    return {
        "sample_covariance": covariance.tolist(),
        "coordinate_standard_error_of_mean": standard_error.tolist(),
    }


def estimate_landscape(
    manifest: ACL004Manifest, landscape: ACL004Landscape
) -> dict[str, Any]:
    """Estimate one conditional mean; sampled code does not call the comparator."""
    analytic = analytic_finite_lambda_direction(
        landscape.state,
        sample_count=manifest.sample_count,
        parent_count=manifest.parent_count,
        quadrature_order=manifest.quadrature_order,
    )
    rng = np.random.Generator(np.random.PCG64(landscape.seed))
    chunks: list[dict[str, Any]] = []
    h1_mean: list[float] = []
    h1_covariance: list[float] = []
    history = []
    converged = False
    stopped = manifest.replication_schedule[-1]
    checkpoint_set = set(manifest.replication_schedule)
    generated = 0
    while generated < manifest.replication_schedule[-1]:
        shadows = sample_rank_mu_shadows(
            landscape.state,
            sample_count=manifest.sample_count,
            parent_count=manifest.parent_count,
            replications=manifest.chunk_size,
            rng=rng,
            batch_size=manifest.chunk_size,
        )
        direction_sum = np.sum(shadows, axis=0)
        outer_sum = shadows.T @ shadows
        chunks.append(
            {
                "chunk_index": len(chunks),
                "count": manifest.chunk_size,
                "direction_sum": direction_sum.tolist(),
                "direction_outer_sum": outer_sum.tolist(),
            }
        )
        if generated < manifest.h1_shadow_count:
            remaining = manifest.h1_shadow_count - generated
            mean_cosines, covariance_cosines = _many_block_cosines(
                landscape.state, shadows[:remaining], analytic
            )
            h1_mean.extend(mean_cosines)
            h1_covariance.extend(covariance_cosines)
        generated += manifest.chunk_size
        if generated not in checkpoint_set:
            continue
        half_chunk_count = len(chunks) // 2
        first_mean = _mean_from_chunks(chunks[:half_chunk_count])
        second_mean = _mean_from_chunks(chunks[half_chunk_count:])
        half_cosines = fisher_block_cosines(landscape.state, first_mean, second_mean)
        history.append(
            {
                "replications": generated,
                "first_half_mean": first_mean.tolist(),
                "second_half_mean": second_mean.tolist(),
                "half_fisher_cosines": half_cosines,
            }
        )
        if (
            half_cosines["mean"] >= manifest.half_cosine_min
            and half_cosines["covariance"] >= manifest.half_cosine_min
        ):
            converged = True
            stopped = generated
            break
    final_mean = _mean_from_chunks(chunks)
    block_cosines = fisher_block_cosines(landscape.state, final_mean, analytic)
    return {
        "landscape_id": landscape.identifier,
        "seed": landscape.seed,
        "converged": converged,
        "stopped_replications": stopped,
        "analytic_direction": analytic.tolist(),
        "stopped_mean_direction": final_mean.tolist(),
        "final_cosines": {
            **block_cosines,
            "joint": _joint_fisher_cosine(landscape.state, final_mean, analytic),
        },
        "checkpoint_history": history,
        "shadow_chunks": chunks,
        "shadow_uncertainty": _uncertainty_from_chunks(chunks),
        "rng_state_after": rng.bit_generator.state,
        "h1": {
            "shadow_count": len(h1_mean),
            "mean_cosines": h1_mean,
            "covariance_cosines": h1_covariance,
        },
    }


def analyze_landscape_results(
    manifest: ACL004Manifest, results: list[dict[str, Any]]
) -> dict[str, Any]:
    if {row["landscape_id"] for row in results} != {
        landscape.identifier for landscape in manifest.landscapes
    }:
        raise ValueError("ACL-004 result landscape IDs differ from manifest")
    convergence = all(row["converged"] for row in results)
    mean_minimum = min(row["final_cosines"]["mean"] for row in results)
    covariance_minimum = min(row["final_cosines"]["covariance"] for row in results)
    if not convergence:
        verdict = "INCONCLUSIVE"
    elif mean_minimum >= manifest.h2_cosine_min and covariance_minimum >= manifest.h2_cosine_min:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    h1_rows = []
    for row in results:
        for block in ("mean", "covariance"):
            values = row["h1"][f"{block}_cosines"]
            h1_rows.append(
                {
                    "landscape_id": row["landscape_id"],
                    "block": block,
                    "count": len(values),
                    "q10": type7_quantile(values, 0.1),
                    "median": type7_quantile(values, 0.5),
                    "q90": type7_quantile(values, 0.9),
                    "fraction_positive": sum(value > 0.0 for value in values) / len(values),
                }
            )
    return {
        "experiment_id": manifest.experiment_id,
        "h2_verdict": verdict,
        "all_landscapes_converged": convergence,
        "convergence_threshold": manifest.half_cosine_min,
        "h2_threshold": manifest.h2_cosine_min,
        "mean_block_minimum_fisher_cosine": mean_minimum,
        "covariance_block_minimum_fisher_cosine": covariance_minimum,
        "joint_cosine_gating": False,
        "landscape_results": [
            {
                "landscape_id": row["landscape_id"],
                "converged": row["converged"],
                "stopped_replications": row["stopped_replications"],
                "final_cosines": row["final_cosines"],
            }
            for row in results
        ],
        "h1_descriptive": h1_rows,
    }


def validate_preregistration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if (
        lock.get("experiment_id") != "ACL-004"
        or lock.get("kind") != "preregistration-bundle-lock"
        or lock.get("outcomes_generated") is not False
        or set(lock.get("files", {})) != ACL004_LOCKED_FILES
    ):
        raise ValueError("ACL-004 lock must contain the exact frozen file set")
    actual_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if actual_files != ACL004_LOCKED_FILES | {"LOCK.json"}:
        raise ValueError("ACL-004 bundle must have exact frozen directory contents")
    manifest = load_manifest(bundle / "manifest.json")
    try:
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-004 analytic registry") from error
    computed = build_analytic_registry(manifest)
    if locked_registry != computed:
        raise ValueError("ACL-004 analytic registry does not match clean recomputation")
    return {
        "schema_version": 1,
        "experiment_id": "ACL-004",
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "landscape_count": len(manifest.landscapes),
        "sample_count": manifest.sample_count,
        "parent_count": manifest.parent_count,
        "replication_schedule": list(manifest.replication_schedule),
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
    canonical = (repo / "evidence" / f"ACL-004-confirmatory-{approved_sha}.json").resolve()
    if requested != canonical:
        raise ValueError("ACL-004 output must equal the SHA-derived canonical evidence path")
    current_sha, dirty = git_execution_state(repo)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        worktree_dirty=dirty,
        output_path=canonical,
    )
    validation = validate_preregistration_bundle(bundle_path)
    manifest = load_manifest(Path(bundle_path) / "manifest.json")
    results = [estimate_landscape(manifest, landscape) for landscape in manifest.landscapes]
    for result in results:
        reproduced = reproduce_stopped_mean(result)
        if not np.array_equal(reproduced, np.asarray(result["stopped_mean_direction"])):
            raise FloatingPointError("ACL-004 chunk statistics do not reproduce stopped mean")
    analysis = analyze_landscape_results(manifest, results)
    payload = {
        "schema_version": 1,
        "experiment_id": "ACL-004",
        "kind": "confirmatory-gaussian-finite-lambda-conditional-mean",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "randomness": "PCG64 independent landscape streams",
        "target_refit": False,
        "lambda_scaling_studied": False,
        "benchmark_scope": manifest.benchmark_scope,
        "inference_scope": manifest.inference_scope,
        "transport_scope": manifest.transport_scope,
        "frozen_design": manifest.raw,
        "landscape_results": results,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(canonical, payload)
