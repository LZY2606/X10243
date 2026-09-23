"""Event-sourced replay engine.

Immutable materials and mutable decisions are separate layers. Replaying the
event log reconstructs derived results (integration, blank-corrected
integration, binding fits) with content-addressed stable ids. Reverting to an
old event restores the dependency versions and conclusions that existed then,
never overwriting history with newest numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .binding import fit_models
from .integrate import apply_blank_correction, integrate
from .models import Event, Material, stable_id
from .parser import parse_material

DEFAULT_TS = "2026-01-01T00:00:00+09:00"


@dataclass
class DecisionRecord:
    event_id: str
    seq: int
    kind: str
    payload: dict[str, Any]
    basis: dict[str, Any]


@dataclass
class BranchState:
    branch: str
    fork_seq: int = 0
    materials: dict[str, dict[int, Material]] = field(default_factory=dict)
    heads: dict[str, int] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    integration: Any = None
    corrected: Any = None
    fits: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    head_seq: int = 0


def make_event(seq, branch, kind, payload, parents=None, basis=None,
               timestamp=DEFAULT_TS, event_id=None) -> Event:
    payload = _jsonable(payload)
    eid = event_id or stable_id("evt", [seq, branch, kind, payload,
                                        sorted(parents or []), basis or {}])
    return Event(event_id=eid, seq=seq, branch=branch, kind=kind,
                 timestamp=timestamp, payload=payload,
                 parents=parents or [], basis=basis or {})


def material_event(seq, branch, doc, parents=None, basis=None,
                   timestamp=DEFAULT_TS, event_id=None) -> Event:
    return make_event(seq, branch, "material_committed", {"doc": doc},
                      parents=parents, basis=basis, timestamp=timestamp,
                      event_id=event_id)


def decision_event(seq, branch, decision, payload=None, parents=None, basis=None,
                   timestamp=DEFAULT_TS, event_id=None) -> Event:
    return make_event(seq, branch, "decision", {"decision": decision, **(payload or {})},
                      parents=parents, basis=basis, timestamp=timestamp,
                      event_id=event_id)


def _jsonable(obj):
    import json
    return json.loads(json.dumps(obj, ensure_ascii=False, default=lambda o: o.to_dict()))


class Replayer:
    def __init__(self, events: list[Event]):
        self.events = sorted(events, key=lambda e: (e.seq, e.event_id))
        self.forks: dict[str, int] = {"main": 0}
        for ev in self.events:
            if ev.branch != "main":
                self.forks.setdefault(
                    ev.branch,
                    int(ev.payload.get("fork_from_seq",
                                       ev.basis.get("fork_from_seq", 0))))

    def branches(self) -> dict[str, int]:
        return dict(self.forks)

    def branch_event_ids(self, branch: str, at_seq: int | None = None) -> list[str]:
        return [e.event_id for e in self._branch_events(branch, at_seq)]

    def _branch_events(self, branch: str, at_seq: int | None) -> list[Event]:
        fork = self.forks.get(branch, 0)
        out = []
        for ev in self.events:
            if at_seq is not None and ev.seq > at_seq:
                continue
            if ev.branch == "main":
                if branch == "main" or ev.seq <= fork:
                    out.append(ev)
            elif ev.branch == branch:
                out.append(ev)
        return sorted(out, key=lambda e: (e.seq, e.event_id))

    def state(self, branch="main", at_seq: int | None = None) -> BranchState:
        st = BranchState(branch=branch, fork_seq=self.forks.get(branch, 0))
        evs = self._branch_events(branch, at_seq)
        for ev in evs:
            self._apply(ev, st)
        st.head_seq = max([e.seq for e in evs] or [0])
        return st

    def _apply(self, ev: Event, st: BranchState):
        if ev.kind == "material_committed":
            parsed = parse_material(ev.payload["doc"])
            declared = int(ev.payload["doc"].get("version", 0))
            prior_head = st.heads.get(parsed.material_type)
            prior_id = None
            if prior_head is not None:
                prior_id = st.materials[parsed.material_type][prior_head].material_id
            version = declared or ((prior_head or 0) + 1)
            mat = Material(material_id=parsed.material_id,
                           material_type=parsed.material_type, version=version,
                           raw_representation=parsed.raw_representation,
                           diagnostics=parsed.diagnostics, supersedes=prior_id)
            st.materials.setdefault(mat.material_type, {})[version] = mat
            st.heads[mat.material_type] = version
            self._recompute(st, changed=mat.material_type)
        elif ev.kind == "decision":
            st.decisions.append(DecisionRecord(
                event_id=ev.event_id, seq=ev.seq,
                kind=ev.payload["decision"],
                payload={k: v for k, v in ev.payload.items() if k != "decision"},
                basis=ev.basis))
            if ev.payload["decision"] == "create_branch":
                st.fork_note = ev.payload.get("fork_from_seq", ev.seq)
            self._recompute(st, changed="decision")

    def _decision(self, st: BranchState, name: str):
        match = [d for d in st.decisions if d.kind == name]
        return match[-1] if match else None

    def _head(self, st, kind):
        return st.materials[kind][st.heads[kind]] if kind in st.heads else None

    def _recompute(self, st: BranchState, changed: str):
        power, injections = self._head(st, "power"), self._head(st, "injections")
        if power is None or injections is None:
            return
        baseline = self._head(st, "baseline")
        drop = self._decision(st, "drop_first_injection")
        ctx = {"drop_first_injection": bool(drop.payload.get("enabled", True))
               if drop is not None else False}

        if changed in ("power", "injections", "baseline", "blank",
                       "concentration", "conditions", "decision"):
            integ = integrate(power, injections, baseline, decision_ctx=ctx)
            st.integration = integ
            st.results.setdefault(integ.result_id, integ)

        blank = self._head(st, "blank")
        bd = self._decision(st, "blank_rule")
        rule = bd.payload["rule"] if bd is not None else ("by_index" if blank else "none")
        if changed in ("power", "injections", "baseline", "blank",
                       "concentration", "conditions", "decision"):
            corrected = apply_blank_correction(st.integration, blank, rule, [])
            st.corrected = corrected
            st.results.setdefault(corrected.result_id, corrected)

        # Existing fits are preserved and only marked stale: a concentration
        # revision invalidates a model without rewriting its original numbers.
        self._mark_fits_stale(st)

        triggers_fit = ("injections", "conditions", "blank", "concentration",
                        "decision")
        if changed in triggers_fit and all(k in st.heads for k in ("concentration", "conditions")):
            fit = fit_models(st.corrected, injections,
                             self._head(st, "concentration"),
                             self._head(st, "conditions"),
                             blank, rule=rule)
            fit.accepted_candidate = self._accepted(st)
            fit.stale, fit.stale_reasons = self._staleness(st, fit)
            st.fits[fit.result_id] = fit
            st.results.setdefault(fit.result_id, fit)

    def _accepted(self, st: BranchState):
        accepted = None
        for d in st.decisions:
            if d.kind == "accept_candidate":
                accepted = d.payload.get("model")
            elif d.kind == "reject_candidate" and accepted == d.payload.get("model"):
                accepted = None
        return accepted

    def _mark_fits_stale(self, st: BranchState):
        current_integration = st.corrected.result_id if st.corrected else None
        for fit in st.fits.values():
            reasons = []
            conc_head = st.heads.get("concentration")
            if conc_head is not None and fit.concentration_version != conc_head:
                reasons.append("concentration_revision")
            cond_head = st.heads.get("conditions")
            if cond_head is not None and fit.conditions_version != cond_head:
                reasons.append("conditions_revision")
            if current_integration is not None and fit.input_summary.get("integration") != current_integration:
                reasons.append("integration_revision")
            fit.stale = bool(reasons)
            fit.stale_reasons = reasons

    def _staleness(self, st: BranchState, fit):
        reasons = []
        conc_head = st.heads.get("concentration")
        if conc_head is not None and fit.concentration_version != conc_head:
            reasons.append("concentration_revision")
        cond_head = st.heads.get("conditions")
        if cond_head is not None and fit.conditions_version != cond_head:
            reasons.append("conditions_revision")
        if st.corrected is not None and fit.input_summary.get("integration") != st.corrected.result_id:
            reasons.append("integration_revision")
        return bool(reasons), reasons
