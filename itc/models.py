"""Domain data model: immutable materials, events, decisions and results.

Layering rule (see README):

* ``MaterialVersion`` objects are *immutable*. Revising a concentration never
  mutates the old power curve or the old integral; it creates a new version.
* ``Event`` rows are append-only and form the audit trail.
* Computed ``Result`` records cache deterministic engine output keyed by an
  input digest. They carry input summaries, parameter versions and parents.
* ``Decision`` rows are the only mutable-in-intent layer: accept/reject/adjust
  actions a reviewer takes. Conflicting decisions are allowed on separate
  branches; an event never overwrites history.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .diagnostics import Status


# --------------------------------------------------------------------------
# Immutable material versions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gap:
    """A missing interval in a power series (seconds, internal time basis)."""
    start: float
    end: float
    reason: str = "missing samples"

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "reason": self.reason}


@dataclass(frozen=True)
class Baseline:
    """Piecewise-linear baseline declared in power-vs-seconds coordinates."""
    points: tuple[tuple[float, float], ...]   # (t_seconds, power_watts)
    raw_unit_time: str
    raw_unit_power: str

    def power_at(self, t: float) -> float:
        pts = self.points
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return p1
                frac = (t - t0) / (t1 - t0)
                return p0 + frac * (p1 - p0)
        return pts[-1][1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [list(p) for p in self.points],
            "raw_unit_time": self.raw_unit_time,
            "raw_unit_power": self.raw_unit_power,
        }


@dataclass(frozen=True)
class Injection:
    """One scheduled injection. ordinal is 1-based and its stable identity."""
    ordinal: int
    time_raw: str
    volume_raw: str
    time_s: float
    volume_l: float
    time_basis: str = "seconds since experiment start"
    endpoint_s: Optional[float] = None       # reviewer-set integration endpoint
    endpoint_raw: Optional[str] = None

    @property
    def identity(self) -> str:
        return f"injection#{self.ordinal}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "identity": self.identity,
            "time_raw": self.time_raw,
            "volume_raw": self.volume_raw,
            "time_s": self.time_s,
            "volume_l": self.volume_l,
            "time_basis": self.time_basis,
            "endpoint_s": self.endpoint_s,
            "endpoint_raw": self.endpoint_raw,
        }


@dataclass(frozen=True)
class MaterialVersion:
    """An immutable, content-addressed revision of one material kind.

    kind is one of: power, injections, concentration, conditions, baseline,
    blank_power, blank_injections. A blank control is the pair
    blank_power + blank_injections.
    """
    kind: str
    version: int
    source_name: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=lambda: time.time())
    parent_version: Optional[int] = None
    note: str = ""

    @property
    def identity(self) -> str:
        return f"material:{self.kind}:v{self.version}"

    def digest(self) -> str:
        body = json.dumps(
            {"kind": self.kind, "version": self.version, "payload": self.payload,
             "source_name": self.source_name},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "identity": self.identity,
            "source_name": self.source_name,
            "payload": self.payload,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "note": self.note,
            "digest": self.digest(),
        }


# --------------------------------------------------------------------------
# Events (append-only) and reviewer decisions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    seq: int
    branch: str
    kind: str                 # material_imported | decision | branch | note
    material_identity: Optional[str]
    payload: dict[str, Any]
    created_at: float
    parent_seq: Optional[int]
    actor: str = "reviewer"

    @property
    def identity(self) -> str:
        return f"event:{self.seq}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "identity": self.identity,
            "branch": self.branch,
            "kind": self.kind,
            "material_identity": self.material_identity,
            "payload": self.payload,
            "created_at": self.created_at,
            "parent_seq": self.parent_seq,
            "actor": self.actor,
        }


@dataclass(frozen=True)
class Decision:
    """An accept / reject / adjust verdict tied to a specific result version."""
    decision_id: str
    branch: str
    action: str               # accept | reject | adjust
    subject: str              # stable identity, e.g. fit:onesite / injection#2
    result_id: Optional[str]
    params: dict[str, Any]
    rationale: str
    event_seq: int
    supersedes: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.action == "accept"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Cached computed results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    result_id: str
    kind: str                 # integration | blank_match | corrected | fit
    status: Status
    inputs_digest: str
    rule_version: str
    parents: tuple[str, ...]
    values: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]
    created_at: float
    identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "identity": self.identity,
            "kind": self.kind,
            "status": self.status.value,
            "inputs_digest": self.inputs_digest,
            "rule_version": self.rule_version,
            "parents": list(self.parents),
            "values": self.values,
            "diagnostics": list(self.diagnostics),
            "created_at": self.created_at,
        }
