"""Common, JSON-safe instrumentation shared by every experimental world."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy-rich structures to strict JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("instrumentation cannot contain NaN or infinity")
    return value


def rng_fingerprint(rng: np.random.Generator | None) -> str:
    """Hash the complete RNG state without consuming it."""
    if rng is None:
        return "none"
    payload = json.dumps(json_safe(rng.bit_generator.state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def rng_snapshot(rng: np.random.Generator | None) -> dict[str, Any] | None:
    """Copy a generator state into a JSON-safe value without consuming it."""
    if rng is None:
        return None
    return json_safe(rng.bit_generator.state)


@dataclass(frozen=True)
class StepRecord:
    """One fully instrumented transition in a native domain."""

    domain: str
    step: int
    canonical_state_before: list[float]
    canonical_state_after: list[float]
    native_state_before: dict[str, Any]
    native_state_after: dict[str, Any]
    expected_vector_field: list[float]
    analytic_gradient: list[float]
    geometry: dict[str, Any]
    realized_update: list[float]
    step_size: float
    stochastic_error: list[float]
    potentials: dict[str, float]
    regret: dict[str, float]
    constraint_violations: dict[str, Any]
    numerical_guards: list[str]
    rng_fingerprint: str
    rng_state: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class Trajectory:
    """A reproducible run configuration and its ordered transition records."""

    domain: str
    config: dict[str, Any]
    records: list[StepRecord] = field(default_factory=list)

    @property
    def terminal_state(self) -> np.ndarray:
        if not self.records:
            initial = self.config.get("initial_state")
            if initial is None:
                raise ValueError("empty trajectory has no initial_state in its config")
            return np.asarray(initial, dtype=np.float64)
        return np.asarray(self.records[-1].canonical_state_after, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "config": json_safe(self.config),
            "records": [record.to_dict() for record in self.records],
        }
