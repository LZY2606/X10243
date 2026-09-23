"""Domain model: versioned immutable materials, mutable decisions, derived results."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ---- three structurally different failure outcomes -------------------------
DiagnosticClass = Literal["resource_exhausted", "format_corrupt", "semantic_unresolvable"]

MATERIAL_TYPES = ("power", "injections", "concentration", "conditions",
                  "baseline", "blank")


@dataclass
class Diagnostic:
    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    diagnostic_class: DiagnosticClass = "semantic_unresolvable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- immutable, separately versioned materials ------------------------------
@dataclass
class PowerSample:
    time_raw: str
    time_unit: str
    power_raw: str
    power_unit: str
    missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Material:
    """A versioned immutable source material.

    Six kinds are versioned independently: power sampling, injection
    volume/timing, concentration, conditions (temperature + cell volume),
    baseline curve, blank curve.
    """

    material_id: str
    material_type: str
    version: int
    raw_representation: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- mutable decisions ------------------------------------------------------
@dataclass
class Decision:
    decision_id: str
    kind: str            # drop_first_injection | blank_rule | accept | reject | param_adjustment | ...
    payload: dict[str, Any]
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- derived results --------------------------------------------------------
@dataclass
class IntegrationPoint:
    injection_index: int
    window_start_s: float | None
    window_end_s: float | None
    heat_j: float | None
    raw_heat_j: float | None
    status: str          # ok | dropped | blocked_by_gap | endpoint_truncated | overlap_split
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    split_boundary_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationResult:
    result_id: str
    power_version: int
    injections_version: int
    baseline_version: int
    input_summary: dict[str, Any]
    param_version: int
    parents: list[str]
    points: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCandidate:
    model: str           # one_site | two_site | dilution
    params: dict[str, float]
    param_units: dict[str, str]
    sse: float
    aic: float
    equivalent_group: int
    rank: int
    usable_points: int
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FitResult:
    result_id: str
    integration_id: str
    blank_id: str | None
    concentration_version: int
    conditions_version: int
    input_summary: dict[str, Any]
    param_version: int
    parents: list[str]
    candidates: list[dict[str, Any]]
    accepted_candidate: str | None = None
    stale: bool = False
    stale_reasons: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- event log (machine readable, replayable) --------------------------------
@dataclass
class Event:
    event_id: str
    seq: int
    branch: str
    kind: str
    timestamp: str
    payload: dict[str, Any]
    parents: list[str] = field(default_factory=list)
    basis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_hash(obj: Any) -> str:
    """Deterministic content hash used for stable identities."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, obj: Any) -> str:
    return f"{prefix}_{canonical_hash(obj)[:16]}"
