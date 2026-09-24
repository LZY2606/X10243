"""Append-only decision/event engine.

Layering enforced here:

* immutable layer - material revisions (the seven versioned source kinds);
* mutable layer   - decisions stored as append-only events;
* derived layer   - integrations, blank matches and model fits, each recording
                    its input summary, parameter/protocol version and parent
                    dependency revision ids.

Returning to any old event replays the prefix and restores the derived
conclusions that existed then: later recomputation never overwrites history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import integrate, models
from .errors import Diagnostic, SemanticInfeasible
from .materials import (
    MATERIAL_KINDS, MaterialRef, MaterialRevision, dehydrate_revision,
    digest_of, parse_material, rehydrate_revision,
)

PROTOCOL_VERSION = "itc-review/1.0"


@dataclass
class Event:
    event_id: str
    seq: int
    branch: str
    kind: str                       # material | decision | computed
    payload: dict[str, Any]
    parent_ids: list[str]
    label: str = ""
    timestamp: str = ""
    conflict_with: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "seq": self.seq, "branch": self.branch,
            "kind": self.kind, "payload": _dehydrate_payload(self.payload),
            "parent_ids": self.parent_ids, "label": self.label,
            "timestamp": self.timestamp, "conflict_with": self.conflict_with,
        }

    @staticmethod
    def from_dict(obj: dict) -> "Event":
        return Event(
            obj["event_id"], obj["seq"], obj["branch"], obj["kind"],
            _hydrate_payload(obj["payload"]), list(obj["parent_ids"]),
            obj.get("label", ""), obj.get("timestamp", ""),
            obj.get("conflict_with"))


def _dehydrate_payload(payload: Any) -> Any:
    if isinstance(payload, MaterialRevision):
        return {"__material__": True, "rev": dehydrate_revision(payload)}
    if isinstance(payload, dict):
        return {k: _dehydrate_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_dehydrate_payload(v) for v in payload]
    return payload


def _hydrate_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and payload.get("__material__"):
        return rehydrate_revision(payload["rev"])
    if isinstance(payload, dict):
        return {k: _hydrate_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_hydrate_payload(v) for v in payload]
    return payload


@dataclass
class DerivedRecord:
    record_id: str
    kind: str                       # integration | blank_match | model_fit
    run_id: str
    stream: str
    inputs_summary: dict
    parents: list[dict]
    protocol_version: str
    data: dict
    produced_by: str
    invalidations: list[dict] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id, "kind": self.kind, "run_id": self.run_id,
            "stream": self.stream, "inputs_summary": self.inputs_summary,
            "parents": self.parents, "protocol_version": self.protocol_version,
            "data": self.data, "produced_by": self.produced_by,
            "invalidations": self.invalidations, "status": self.status,
        }

    @staticmethod
    def from_dict(obj: dict) -> "DerivedRecord":
        return DerivedRecord(
            obj["record_id"], obj["kind"], obj["run_id"], obj.get("stream", "titration"),
            obj["inputs_summary"], obj["parents"], obj["protocol_version"],
            obj["data"], obj["produced_by"], obj.get("invalidations", []),
            obj.get("status", "ok"))


@dataclass
class ProjectState:
    project_id: str
    title: str
    events: list[Event] = field(default_factory=list)
    # material_id -> run stream -> {version: rev, current: version}
    materials: dict[str, dict[str, dict]] = field(default_factory=dict)
    # run_id -> {"titration": stream, ...} -> run view (which streams exist)
    runs: dict[str, dict] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    derived: list[DerivedRecord] = field(default_factory=list)
    branches: dict[str, list[str]] = field(default_factory=lambda: {"main": []})
    head: dict[str, str] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)

    # -- material access -------------------------------------------------- #
    def current(self, material_id: str, stream: str = "titration"
                ) -> MaterialRevision | None:
        entry = self.materials.get(material_id, {}).get(stream)
        if not entry:
            return None
        return entry["versions"][entry["current"]]

    def heads_for(self, run_id: str, stream: str = "titration") -> dict[str, MaterialRevision]:
        out = {}
        for kind in MATERIAL_KINDS:
            mid = f"{kind}.{run_id}"
            rev = self.current(mid, stream if kind in ("power", "injections") else "titration")
            if rev is not None:
                out[kind] = rev
        return out

    def branch_event_ids(self, branch: str) -> list[str]:
        return list(self.branches.get(branch, []))


# --------------------------------------------------------------------------- #
# Event creation / append
# --------------------------------------------------------------------------- #

def new_project(project_id: str, title: str) -> ProjectState:
    return ProjectState(project_id=project_id, title=title)


def stream_for(kind: str, run_id: str, stream: str | None) -> str:
    return stream if (kind in ("power", "injections") and stream) else "titration"


def make_material_event(state: ProjectState, kind: str, raw: dict, *, run_id: str,
                        source: str, branch: str = "main", stream: str | None = None,
                        label: str = "", timestamp: str = "",
                        force_version: int | None = None) -> tuple[Event, MaterialRevision]:
    use_stream = stream_for(kind, run_id, stream)
    mid = f"{kind}.{run_id}"
    entry = state.materials.setdefault(mid, {}).setdefault(
        use_stream, {"versions": {}, "current": 0})
    version = force_version if force_version is not None else entry["current"] + 1
    rev = parse_material(kind, raw, run_id=run_id, source=source, version=version)
    if rev.revision_id in [v.revision_id for v in entry["versions"].values()]:
        # identical content: stable identity means no new revision
        existing = next(v for v in entry["versions"].values()
                        if v.revision_id == rev.revision_id)
        return _material_event(state, existing, branch, label, timestamp), existing
    event = _material_event(state, rev, branch, label, timestamp, use_stream)
    return event, rev


def _material_event(state: ProjectState, rev: MaterialRevision, branch: str,
                    label: str, timestamp: str, use_stream: str = "titration") -> Event:
    seq = len(state.events) + 1
    parent_ids = [state.head[branch]] if branch in state.head else []
    body = {"material": rev, "stream": use_stream}
    event_id = f"E{seq:04d}:{rev.kind}:{digest_of(rev.revision_id + '|' + '|'.join(parent_ids))}"
    return Event(event_id, seq, branch, "material", body, parent_ids,
                 label or f"{rev.kind} {rev.run_id} v{rev.version}", timestamp)


def make_decision_event(state: ProjectState, decision: dict, *, branch: str = "main",
                        label: str = "", timestamp: str = "",
                        conflict_with: str | None = None) -> Event:
    seq = len(state.events) + 1
    parent_ids = [state.head[branch]] if branch in state.head else []
    payload = {"decision": decision}
    body_json = json.dumps(decision, sort_keys=True, default=str)
    event_id = f"E{seq:04d}:decision:{digest_of(body_json + '|' + '|'.join(parent_ids))}"
    return Event(event_id, seq, branch, "decision", payload, parent_ids,
                 label or decision.get("action", "decision"), timestamp, conflict_with)


def make_computed_event(state: ProjectState, produced: list[DerivedRecord],
                        invalidated: list[str], *, branch: str, trigger: str,
                        timestamp: str = "") -> Event:
    seq = len(state.events) + 1
    parent_ids = [state.head[branch]] if branch in state.head else []
    payload = {"record_ids": [r.record_id for r in produced],
               "invalidated": invalidated, "trigger": trigger}
    event_id = (f"E{seq:04d}:computed:"
                f"{digest_of(json.dumps(payload, sort_keys=True) + '|' + '|'.join(parent_ids))}")
    return Event(event_id, seq, branch, "computed", payload, parent_ids,
                 f"derived recompute ({trigger})", timestamp)


# --------------------------------------------------------------------------- #
# Apply / replay
# --------------------------------------------------------------------------- #

def apply_event(state: ProjectState, event: Event, *, recompute: bool = True,
                budget: int = 65536) -> list[DerivedRecord]:
    """Append one event and advance branch head.

    Material events store a new immutable revision; decision events append a
    decision (manual window override, accept/reject model, endpoint choice...).
    Derived records are then rebuilt deterministically for every run/stream.
    Old derived records are retained with ``status='superseded'`` so history is
    never overwritten.
    """
    state.events.append(event)
    state.branches.setdefault(event.branch, [])
    if event.event_id not in state.branches[event.branch]:
        state.branches[event.branch].append(event.event_id)
    state.head[event.branch] = event.event_id

    invalidated: list[str] = []
    if event.kind == "material":
        rev: MaterialRevision = event.payload["material"]
        use_stream = event.payload.get("stream", "titration")
        entry = state.materials.setdefault(rev.material_id, {}).setdefault(
            use_stream, {"versions": {}, "current": 0})
        entry["versions"][rev.version] = rev
        entry["current"] = rev.version
        state.runs.setdefault(rev.run_id, {"streams": set()})
        if use_stream != "titration":
            state.runs[rev.run_id]["streams"].add(use_stream)
        invalidated = _invalidate_for(state, rev.kind, rev.run_id, use_stream)
    elif event.kind == "decision":
        decision = event.payload["decision"]
        state.decisions.append({"event_id": event.event_id,
                                "branch": event.branch, **decision,
                                "conflict_with": event.conflict_with})
        invalidated = _invalidate_for_decision(state, decision)

    produced: list[DerivedRecord] = []
    if recompute and event.kind in ("material", "decision"):
        produced = recompute_all(state, branch=event.branch,
                                 trigger=event.event_id, budget=budget,
                                 invalidated=invalidated)
        comp = make_computed_event(state, produced, invalidated,
                                   branch=event.branch, trigger=event.event_id,
                                   timestamp=event.timestamp)
        # computed event itself does not trigger another recompute
        state.events.append(comp)
        state.branches[event.branch].append(comp.event_id)
        state.head[event.branch] = comp.event_id
    return produced



def _invalidate_for(state: ProjectState, kind: str, run_id: str,
                    stream: str) -> list[str]:
    affected_kinds = {
        "power": ["integration", "blank_match", "model_fit"],
        "injections": ["integration", "blank_match", "model_fit"],
        "baseline": ["integration", "blank_match", "model_fit"],
        "temperature": [],
        "cell_volume": ["integration", "model_fit"],
        "concentration": ["model_fit"],
        "blank_link": ["blank_match", "model_fit"],
    }[kind]
    out = []
    for rec in state.derived:
        if rec.status == "current" and rec.kind in affected_kinds:
            if rec.run_id == run_id and rec.stream == stream:
                out.append(rec.record_id)
            elif rec.kind == "blank_match" and rec.data.get("blank_run") == run_id:
                out.append(rec.record_id)
    return sorted(set(out))


def _invalidate_for_decision(state: ProjectState, decision: dict) -> list[str]:
    action = decision.get("action")
    if action == "window_override":
        kinds = ["integration", "blank_match", "model_fit"]
    elif action in ("accept_model", "reject_model", "parameter_override"):
        kinds = ["model_fit"]
    elif action == "endpoint_decision":
        kinds = ["integration", "blank_match", "model_fit"]
    else:
        kinds = []
    run_id = decision.get("run_id", "")
    out = [r.record_id for r in state.derived
           if r.status == "current" and r.kind in kinds
           and (not run_id or r.run_id == run_id)]
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Derived recomputation
# --------------------------------------------------------------------------- #

def active_decisions(state: ProjectState, branch: str, run_id: str) -> list[dict]:
    ids = set(state.branches.get(branch, []))
    out = []
    for d in state.decisions:
        if d["event_id"] in ids and d.get("run_id", run_id) == run_id:
            out.append(d)
    return out


def _overrides(decisions: list[dict], stream: str) -> dict[int, dict]:
    out = {}
    for d in decisions:
        if d.get("action") == "window_override" and d.get("stream", "titration") == stream:
            out[int(d["ordinal"])] = {
                "window_end_s": d.get("window_end_s"),
                "decided_by": d["event_id"],
            }
    return out


def recompute_all(state: ProjectState, *, branch: str, trigger: str,
                  budget: int, invalidated: list[str]) -> list[DerivedRecord]:
    # mark superseded
    invalid_set = set(invalidated)
    for rec in state.derived:
        if rec.record_id in invalid_set:
            rec.status = "superseded"
            rec.invalidations.append({"by": trigger, "reason": "input_or_decision_changed"})

    produced: list[DerivedRecord] = []
    # gather (run_id, stream) pairs that have power materials
    pairs: set[tuple[str, str]] = set()
    for mid, streams in state.materials.items():
        if not mid.startswith("power."):
            continue
        run_id = mid[len("power."):]
        for use_stream in streams:
            pairs.add((run_id, use_stream))

    integrated: dict[tuple[str, str], dict] = {}
    for run_id, use_stream in sorted(pairs):
        built = build_integration(state, run_id, use_stream, branch, trigger)
        if built is not None:
            integrated[(run_id, use_stream)] = built
            produced.append(built["integration_record"])

    # blank matches: a titration run with a blank_link integrates the blank
    blank_by_run: dict[str, dict] = {}
    for run_id, use_stream in sorted(pairs):
        if use_stream != "titration":
            continue
        link = state.current(f"blank_link.{run_id}", "titration")
        if link is None:
            continue
        blank_run = link.data["blank_run"]
        if (blank_run, "titration") not in integrated:
            built = build_integration(state, blank_run, "titration", branch, trigger)
            if built is not None:
                integrated[(blank_run, "titration")] = built
                produced.append(built["integration_record"])
        blank_by_run[run_id] = link

    for run_id, link in sorted(blank_by_run.items()):
        sample = integrated.get((run_id, "titration"))
        blank = integrated.get((link.data["blank_run"], "titration"))
        rec = build_blank_match(state, run_id, link, sample, blank, trigger)
        produced.append(rec)

    # models are fitted per titration run after blank matches exist
    for run_id in sorted(blank_by_run):
        rec = build_model_fit(state, run_id, branch, trigger, budget)
        if rec is not None:
            produced.append(rec)
    return produced


def build_integration(state: ProjectState, run_id: str, use_stream: str,
                      branch: str, trigger: str) -> dict | None:
    power = state.current(f"power.{run_id}", use_stream)
    injections = state.current(f"injections.{run_id}", use_stream)
    baseline = state.current(f"baseline.{run_id}", "titration")
    if power is None or injections is None or baseline is None:
        return None
    arrays, inj_rows = integrate.extract_run(power, injections, baseline)
    decisions = active_decisions(state, branch, run_id)
    points = integrate.integrate_run(
        arrays, inj_rows, baseline, overrides=_overrides(decisions, use_stream))
    summary = {
        "run_id": run_id, "stream": use_stream,
        "n_samples": int(arrays.time_s.size),
        "n_injections": len(points),
        "time_reference": arrays.time_reference,
        "energy_unit": "cal",
        "power_diagnostics": [d.to_dict() for d in arrays.power_diagnostics],
    }
    parents = [power.ref().to_dict(), injections.ref().to_dict(),
               baseline.ref().to_dict()]
    data = {"points": [p.to_dict() for p in points],
            "trace": _trace_summary(arrays)}
    record_id = _rid("integration", run_id, use_stream, parents, data)
    rec = DerivedRecord(record_id, "integration", run_id, use_stream, summary,
                        parents, PROTOCOL_VERSION, data, trigger)
    state.derived.append(rec)
    return {"integration_record": rec, "points": points, "arrays": arrays}


def build_blank_match(state: ProjectState, sample_run: str, link,
                      sample_built: dict | None, blank_built: dict | None,
                      trigger: str) -> DerivedRecord:
    if sample_built is None:
        sample_points = []
    else:
        sample_points = sample_built["points"]
    blank_points = blank_built["points"] if blank_built else []
    matches = integrate.match_blanks(sample_points, blank_points, link)
    rows = integrate.corrected_heats(sample_points, matches)
    data = {"blank_run": link.data["blank_run"], "rule": link.data["rule"],
            "matches": matches, "corrected_heats": rows}
    parents = [link.ref()]
    if sample_built:
        parents.append(MaterialRef("__derived__", 1,
                                   sample_built["integration_record"].record_id))
    if blank_built:
        parents.append(MaterialRef("__derived__", 1,
                                   blank_built["integration_record"].record_id))
    parent_dicts = [p.to_dict() for p in parents]
    summary = {"sample_run": sample_run, "blank_run": link.data["blank_run"],
                "rule": link.data["rule"], "n": len(rows),
                "missing": [r["ordinal"] for r in rows
                            if r["blank_status"] == "missing_counterpart"]}
    record_id = _rid("blank_match", sample_run, "titration", parent_dicts, data)
    rec = DerivedRecord(record_id, "blank_match", sample_run, "titration",
                        summary, parent_dicts, PROTOCOL_VERSION, data, trigger)
    state.derived.append(rec)
    return rec


def build_model_fit(state: ProjectState, run_id: str, branch: str,
                    trigger: str, budget: int) -> DerivedRecord | None:
    link = state.current(f"blank_link.{run_id}", "titration")
    if link is None:
        return None
    blank_rec = _latest_current(state, "blank_match", run_id)
    conc = state.current(f"concentration.{run_id}", "titration")
    cell_vol = state.current(f"cell_volume.{run_id}", "titration")
    temp = state.current(f"temperature.{run_id}", "titration")
    if blank_rec is None or conc is None or cell_vol is None:
        return None
    volume_l = float(
        __import__("itc.units", fromlist=["require_number"]).require_number(
            cell_vol.data["cell_volume"], "volume"))
    outcome = {"kind": "ok", "message": "", "diagnostics": []}
    titration = None
    try:
        titration = models.build_titration(
            blank_rec.data["corrected_heats"], conc, volume_l, require_blank=True)
    except Exception as exc:  # SemanticInfeasible carries structured diags
        outcome = {"kind": getattr(exc, "outcome", "semantic_infeasible"),
                   "message": str(exc),
                   "diagnostics": [d.to_dict() for d in getattr(exc, "diagnostics", [])]}
    fits: list[dict] = []
    if titration is not None:
        try:
            for fit in models.fit_all(titration, budget=budget):
                fits.append(fit.to_dict())
        except Exception as exc:
            outcome = {"kind": getattr(exc, "outcome", "budget_exhausted"),
                       "message": str(exc),
                       "diagnostics": [d.to_dict() for d in getattr(exc, "diagnostics", [])]}
    # attach reviewer decisions about models
    decisions = [d for d in active_decisions(state, branch, run_id)
                 if d.get("action") in ("accept_model", "reject_model",
                                        "parameter_override")]
    data = {"fits": fits, "outcome": outcome, "excluded":
            titration.excluded if titration is not None else [],
            "decisions": _decision_views(decisions),
            "units": {"heat_per_mole": "cal/mol", "concentration": "M",
                      "temperature_K": None if temp is None else
                      __import__("itc.units", fromlist=["normalise"]).normalise(
                          temp.data["temperature"], "temperature")[0]}}
    parents_dicts = [link.ref().to_dict(), conc.ref().to_dict(),
                     cell_vol.ref().to_dict()]
    if temp is not None:
        parents_dicts.append(temp.ref().to_dict())
    parents_dicts.append({"material_id": "__derived__", "version": 1,
                          "revision_id": blank_rec.record_id})
    summary = {"run_id": run_id, "n_fits": len(fits), "outcome": outcome["kind"],
               "excluded": data["excluded"]}
    record_id = _rid("model_fit", run_id, "titration", parents_dicts, data)
    rec = DerivedRecord(record_id, "model_fit", run_id, "titration", summary,
                        parents_dicts, PROTOCOL_VERSION, data, trigger,
                        status="current" if outcome["kind"] == "ok" else "diagnostic")
    state.derived.append(rec)
    return rec


def _decision_views(decisions: list[dict]) -> list[dict]:
    return [{"event_id": d["event_id"], "action": d["action"],
             "model": d.get("model"), "params": d.get("params"),
             "reason": d.get("reason", ""), "conflict_with": d.get("conflict_with")}
            for d in decisions]


def _latest_current(state: ProjectState, kind: str, run_id: str) -> DerivedRecord | None:
    for rec in reversed(state.derived):
        if rec.kind == kind and rec.run_id == run_id and rec.status in ("current", "diagnostic", "ok"):
            return rec
    return None


def _trace_summary(arrays) -> dict:
    t = arrays.time_s
    if t.size == 0:
        return {"t0_s": None, "t1_s": None, "gaps_s": arrays.gaps_s}
    return {"t0_s": float(t[0]), "t1_s": float(t[-1]),
            "gaps_s": [list(g) for g in arrays.gaps_s],
            "time_reference": arrays.time_reference}


def _rid(kind: str, run_id: str, stream: str, parents: list, data: dict) -> str:
    basis = canonical_basis({"kind": kind, "run_id": run_id, "stream": stream,
                             "parents": parents,
                             "data": _jsonable(data)})
    return f"{kind}.{run_id}.{stream}::{digest_of(basis)}"


def canonical_basis(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ""), default=str,
                      ensure_ascii=False)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, MaterialRef):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


# --------------------------------------------------------------------------- #
# Branching and historical checkout
# --------------------------------------------------------------------------- #

def branch_at(state: ProjectState, parent_branch: str, new_branch: str,
              *, at_event: str | None = None) -> None:
    if new_branch in state.branches:
        raise SemanticInfeasible(
            f"branch {new_branch!r} already exists",
            [Diagnostic("branch:exists", "duplicate branch", "branch", new_branch)])
    source = list(state.branches[parent_branch])
    if at_event is not None:
        if at_event not in source:
            raise SemanticInfeasible(
                f"event {at_event} not on branch {parent_branch}",
                [Diagnostic("branch:unknown_event", "event not on branch",
                            "at_event", at_event)])
        source = source[:source.index(at_event) + 1]
    state.branches[new_branch] = source
    state.head[new_branch] = source[-1] if source else state.head.get(parent_branch, "")


def checkout_prefix(state: ProjectState, branch: str, at_event: str) -> "ProjectState":
    """Replay a branch prefix to produce the state visible at ``at_event``.

    The original state is untouched; the returned copy carries exactly the
    material revisions, decisions and derived records produced up to that
    event.  Historical derived results are reconstructed, not replaced by
    later computations.
    """
    ids = list(state.branches[branch])
    if at_event not in ids:
        raise SemanticInfeasible(
            f"event {at_event} not on branch {branch}",
            [Diagnostic("checkout:unknown_event", "event not on branch",
                        "at_event", at_event)])
    prefix_ids = ids[:ids.index(at_event) + 1]
    by_id = {e.event_id: e for e in state.events}
    clone = new_project(state.project_id, state.title)
    for eid in prefix_ids:
        event = Event.from_dict(by_id[eid].to_dict())
        apply_event(clone, event)
    return clone
