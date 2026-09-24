"""Replay an event stream into a visible project state.

Immutable materials and reviewable decisions live in separate layers:
material_edit events append new versions; integration/fit events record input
summaries + parent dependency ids and are marked stale when their inputs gain a
new version. No recomputed result ever overwrites an old one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .materials import MaterialVersion

DECISION_TYPES = {
    "discard_shots", "blank_rule", "integration_endpoints",
    "manual_decision", "parameter_override",
}
MATERIAL_TYPES = {
    "material_edit": None,
}


@dataclass
class IntegrationRec:
    event_id: str
    seq: int
    settings: dict
    input_summary: dict
    parents: dict
    shots: list[dict]
    diagnostics: list[dict]
    ts: float


@dataclass
class FitRec:
    event_id: str
    seq: int
    integration_id: str
    settings: dict
    input_summary: dict
    parents: dict
    candidates: list[dict]
    diagnostics: list[dict]
    ts: float


@dataclass
class State:
    project_id: str
    name: str
    materials: dict[str, list[MaterialVersion]] = field(default_factory=dict)
    material_event_ids: dict[str, list[str]] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    integrations: list[IntegrationRec] = field(default_factory=list)
    fits: list[FitRec] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    branch_heads: dict[str, str] = field(default_factory=dict)

    def material(self, kind: str) -> MaterialVersion | None:
        versions = self.materials.get(kind)
        return versions[-1] if versions else None

    def material_version(self, kind: str) -> int:
        return self.material(kind).version if self.material(kind) else 0


def replay(events: list[dict], *, project_id: str, name: str = "") -> State:
    state = State(project_id=project_id, name=name)
    for ev in events:
        _apply(state, ev)
    return state


def _apply(state: State, ev: dict) -> None:
    etype, payload = ev["type"], ev["payload"]
    if etype == "project_created":
        state.name = payload.get("name", state.name)
        if payload.get("seed"):
            state.diagnostics.append(
                {"code": "seeded_example", "message": "loaded bundled offline case",
                 "event_id": ev["event_id"]})
    elif etype == "material_edit":
        mv = MaterialVersion.from_dict(payload["material"])
        versions = state.materials.setdefault(mv.kind, [])
        # identical re-import is a no-op (stable equivalence); a new distinct
        # payload always appends a new version.
        if versions and _same_material(versions[-1], mv):
            return
        expected = len(versions) + 1
        if mv.version != expected:
            mv = MaterialVersion(
                kind=mv.kind, version=expected, origin=mv.origin, series=mv.series,
                injections=mv.injections, quantity=mv.quantity, unit=mv.unit,
                raw=mv.raw, extra=mv.extra,
            )
        versions.append(mv)
        state.material_event_ids.setdefault(mv.kind, []).append(ev["event_id"])
        if payload.get("diagnostics"):
            for d in payload["diagnostics"]:
                state.diagnostics.append({**d, "material_kind": mv.kind,
                                          "material_version": mv.version})
    elif etype == "import_diagnostics":
        for d in payload.get("diagnostics", []):
            state.diagnostics.append(d)
    elif etype in DECISION_TYPES:
        state.decisions.append(_decision_view(ev))
    elif etype == "integrated":
        state.integrations.append(IntegrationRec(
            event_id=ev["event_id"], seq=ev["seq"], settings=payload["settings"],
            input_summary=payload["input_summary"], parents=payload["parents"],
            shots=payload["shots"], diagnostics=payload.get("diagnostics", []),
            ts=ev["ts"]))
    elif etype == "fitted":
        state.fits.append(FitRec(
            event_id=ev["event_id"], seq=ev["seq"],
            integration_id=payload["integration_id"], settings=payload["settings"],
            input_summary=payload["input_summary"], parents=payload["parents"],
            candidates=payload["candidates"], diagnostics=payload.get("diagnostics", []),
            ts=ev["ts"]))
    elif etype == "review_note":
        state.decisions.append(_decision_view(ev))


def _same_material(a: MaterialVersion, b: MaterialVersion) -> bool:
    return a.to_dict() == b.to_dict()


def _decision_view(ev: dict) -> dict:
    p = ev["payload"]
    return {
        "event_id": ev["event_id"], "seq": ev["seq"], "type": ev["type"],
        "ts": ev["ts"], "rationale": p.get("rationale", ""),
        "basis": p.get("basis", []), "scope": p.get("scope", {}),
        "payload": p,
    }
