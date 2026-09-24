"""Replay an event chain into a fully-derived review state.

Replay genuinely runs the domain computations (integration + model fitting),
so every derived artifact carries:

* ``input_summary`` - the data actually used,
* ``parents``       - exact material version ids/digests + rule versions,
* ``stale``         - whether a newer material invalidates it.

Artifacts are keyed by the hash of their inputs; the same inputs always yield
the same id and ordering, so history is never overwritten by later results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import integration as integ
from . import materials as mats
from . import thermo
from .ident import canonical, digest, make_id

RULE_VERSION = "rules-itc-v1"


@dataclass
class BranchState:
    branch_id: str
    stream_versions: dict[str, Any] = field(default_factory=dict)
    stream_heads: dict[str, int] = field(default_factory=dict)
    discarded: set[int] = field(default_factory=set)
    blank_rule: str | None = None
    endpoint_overrides: dict[int, dict[str, float]] = field(default_factory=dict)
    integrations: list[dict[str, Any]] = field(default_factory=list)
    integration_art_id: str | None = None
    fits: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.stream_versions:
            self.stream_versions = {k: None for k in mats.STREAM_KINDS}
            self.stream_heads = {k: 0 for k in mats.STREAM_KINDS}


class MaterialStore:
    """In-memory material versions, populated as events are replayed."""

    def __init__(self):
        self.versions: dict[str, list[mats.MaterialVersion]] = {}
        self.by_id: dict[str, mats.MaterialVersion] = {}

    def add(self, mv: mats.MaterialVersion) -> None:
        self.versions.setdefault(mv.case_id + "::" + mv.kind, []).append(mv)
        self.by_id[mv.id] = mv

    def history(self, case_id: str, kind: str) -> list[mats.MaterialVersion]:
        return self.versions.get(case_id + "::" + kind, [])

    def at(self, version_id: str | None) -> mats.MaterialVersion | None:
        return self.by_id.get(version_id) if version_id else None


def _qval(node):
    return integ._qval(node)


def compute_integrations(state: BranchState, store: MaterialStore) -> dict[str, Any]:
    """Integrate every injection against the branch's current material heads."""
    mv = {k: store.at(state.stream_versions.get(k)) for k in mats.STREAM_KINDS}
    power = mv["power"].content if mv["power"] else None
    inj = mv["injections"].content if mv["injections"] else None
    base = mv["baseline"].content if mv["baseline"] else None
    blank = mv["blank"].content if mv["blank"] else None
    if power is None or inj is None:
        return {"status": "missing_inputs", "results": [],
                "diagnostics": [{"code": "missing_inputs",
                                 "message": "power and injections are required",
                                 "location": "/materials", "severity": "error"}],
                "parents": _parents(mv, state.blank_rule)}
    results = integ.integrate_all(power, inj, base)
    for r in results:
        if r["index"] in state.discarded:
            r["status"] = "discarded_first" if r["index"] == min(
                x.get("index", 10**9) for x in inj.get("rows", [])
            ) else "discarded"
            r["discarded"] = True
    # reviewer endpoint adjustments clip windows (never fabricate data)
    for r in results:
        ov = state.endpoint_overrides.get(r["index"])
        if ov and r.get("window"):
            r["window"] = dict(r["window"], reviewer_end=ov.get("end"),
                               reviewer_start=ov.get("start"))
    blank_power = blank.get("power") if blank else None
    results = integ.subtract_blanks(results, blank, blank_power, None,
                                    state.blank_rule)
    parents = _parents(mv, state.blank_rule)
    summary = _integration_summary(results)
    art_id = make_id("intg", state.branch_id, digest(
        [parents, summary, state.discarded and sorted(state.discarded),
         state.endpoint_overrides, RULE_VERSION], length=24), length=24)
    return {"id": art_id, "status": "ok", "results": results,
            "parents": parents, "input_summary": summary,
            "rule_version": RULE_VERSION,
            "blank_rule": state.blank_rule,
            "discarded_indices": sorted(state.discarded)}


def _parents(mv: dict, blank_rule: str | None) -> dict[str, Any]:
    out = {}
    for kind, v in mv.items():
        out[kind] = ({"id": v.id, "version": v.version, "digest": v.digest}
                     if v else None)
    out["blank_rule"] = blank_rule
    out["rule_version"] = RULE_VERSION
    return out


def _integration_summary(results: list[dict]) -> list[dict]:
    return [{"index": r["index"], "status": r["status"],
             "q_baseline_uJ": r.get("q_baseline_uJ"),
             "q_net_uJ": r.get("q_net_uJ"),
             "window_end": (r.get("window") or {}).get("end"),
             "overlaps_next": r.get("overlaps_next")} for r in results]


def compute_fits(state: BranchState, store: MaterialStore,
                 integration: dict[str, Any], *, budget: int = thermo.DEFAULT_BUDGET
                 ) -> dict[str, Any]:
    mv = {k: store.at(state.stream_versions.get(k)) for k in mats.STREAM_KINDS}
    conc = mv["concentrations"].content if mv["concentrations"] else {}
    cellv = mv["cell_volume"].content if mv["cell_volume"] else {}
    inj = mv["injections"].content if mv["injections"] else {}
    m0 = _qval(conc.get("cell"))
    v0 = _qval(cellv.get("value"))
    rows = inj.get("rows", []) if inj else []
    if m0 is None or v0 is None:
        return {"status": "semantic_unresolvable", "candidates": [],
                "reason": "cell concentration and cell volume are required",
                "diagnostics": [
                    {"code": "missing_concentration" if m0 is None
                     else "missing_cell_volume",
                     "message": "needed for binding model normalization",
                     "location": "/concentrations/cell" if m0 is None
                     else "/cell_volume", "severity": "error"}]}
    res = integration.get("results", [])
    q, dv, idx = [], [], []
    for r in res:
        if r.get("discarded") or r["index"] in state.discarded:
            continue
        qi = r.get("q_net_uJ")
        if qi is None:
            qi = r.get("q_baseline_uJ")
        if qi is None:
            continue
        row = next((x for x in rows if x.get("index") == r["index"]), None)
        volume = _qval((row or {}).get("volume"))
        if volume is None:
            continue
        q.append(qi)
        dv.append(volume)
        idx.append(r["index"])
    if len(q) < 2:
        return {"status": "semantic_unresolvable", "candidates": [],
                "reason": "fewer than two quantified injections",
                "diagnostics": [{"code": "too_few_points",
                                 "message": "need >=2 integrated injections",
                                 "location": "/artifacts/integrations",
                                 "severity": "error"}]}
    q_a = np.array(q)
    dv_a = np.array(dv)
    # cumulative ligand concentration in the cell after each injection,
    # accounting for dilution (standard approximation V0 constant)
    x = np.cumsum(dv_a) / v0 * (_qval(conc.get("syringe")) or 0.0)
    fit = thermo.fit_models(q_a, dv_a, x, m0, v0, budget=budget,
                            discard=set())  # discard already removed above
    fit["injection_indices"] = idx
    fit["parents"] = {
        "concentrations": _ref(mv["concentrations"]),
        "cell_volume": _ref(mv["cell_volume"]),
        "injections": _ref(mv["injections"]),
        "integration": integration.get("id"),
        "rule_version": RULE_VERSION}
    fit["input_summary"] = {"q_uJ": q, "dv_uL": dv, "x_uM": x.tolist(),
                            "m0_uM": m0, "v0_uL": v0,
                            "syringe_uM": _qval(conc.get("syringe"))}
    fit_id = make_id("fit", state.branch_id, digest(
        [fit["parents"], fit["input_summary"],
         sorted(c["model"] for c in fit["candidates"]), RULE_VERSION],
        length=24), length=24)
    fit["id"] = fit_id
    return fit


def _ref(v):
    return None if v is None else {"id": v.id, "version": v.version,
                                   "digest": v.digest}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    states: dict[str, BranchState]
    branch_order: list[str]
    store: MaterialStore
    artifacts: dict[str, dict[str, Any]]
    cases: dict[str, dict[str, Any]]
    checkpoints: list[dict[str, Any]]

    def branch(self, branch_id: str) -> BranchState:
        return self.states[branch_id]


def replay(events: list[dict[str, Any]], material_rows: list[dict[str, Any]],
           *, budget: int = thermo.DEFAULT_BUDGET,
           budget_by_event: dict[str, int] | None = None
           ) -> ReplayResult:
    """Replay material versions + ordered events into branch states."""
    store = MaterialStore()
    for row in material_rows:
        mv = mats.MaterialVersion(
            case_id=row["case_id"], kind=row["kind"], version=row["version"],
            content=row["content"], digest=row["digest"], label=row.get("label", ""),
            source_event_id=row.get("source_event_id"))
        store.add(mv)
    states: dict[str, BranchState] = {}
    artifacts: dict[str, dict] = {}
    cases: dict[str, dict] = {}
    checkpoints: list[dict] = []
    order: list[str] = []
    ordered = sorted(events, key=lambda e: (e["seq"], e["branch_seq"],
                                            e["branch_id"]))
    for ev in ordered:
        bid = ev["branch_id"]
        cid = ev["case_id"]
        if ev["kind"] == "case_created":
            cases[cid] = {"id": cid, "title": ev["payload"].get("title", cid),
                          "created_event": ev["id"]}
            continue
        if ev["kind"] == "branch_created":
            parent_branch = ev["payload"].get("parent_branch")
            if parent_branch and parent_branch in states:
                src = states[parent_branch]
                st = BranchState(bid)
                st.stream_versions = dict(src.stream_versions)
                st.stream_heads = dict(src.stream_heads)
                st.discarded = set(src.discarded)
                st.blank_rule = src.blank_rule
                st.endpoint_overrides = dict(src.endpoint_overrides)
            else:
                st = BranchState(bid)
            states[bid] = st
            order.append(bid)
            st.events.append(ev["id"])
            checkpoints.append(_checkpoint(ev, st, artifacts))
            continue
        st = states.get(bid)
        if st is None:
            st = BranchState(bid)
            states[bid] = st
            order.append(bid)
        _apply(ev, st, store, artifacts, cases,
               budget=budget_by_event.get(ev["id"], budget)
               if budget_by_event else budget)
        st.events.append(ev["id"])
        checkpoints.append(_checkpoint(ev, st, artifacts))
    return ReplayResult(states=states, branch_order=order, store=store,
                        artifacts=artifacts, cases=cases,
                        checkpoints=checkpoints)


def _apply(ev: dict, st: BranchState, store: MaterialStore,
           artifacts: dict[str, dict], cases: dict, *, budget: int) -> None:
    p = ev["payload"]
    if ev["kind"] == "material_committed":
        kind = p["kind"]
        mv = store.at(p["material_id"])
        if mv is not None:
            st.stream_versions[kind] = mv.id
            st.stream_heads[kind] = mv.version
        st.integration_art_id = None
        st.fits = []
    elif ev["kind"] == "first_injection_discard":
        action = p.get("action", "discard")
        indices = set(p.get("indices") or [p.get("index")] or [])
        indices.discard(None)
        if action == "discard":
            st.discarded |= indices
        else:
            st.discarded -= indices
    elif ev["kind"] == "blank_rule_set":
        st.blank_rule = p.get("rule") or None
    elif ev["kind"] == "integration_endpoint":
        idx = p["index"]
        if p.get("action") == "reset":
            st.endpoint_overrides.pop(idx, None)
        else:
            st.endpoint_overrides[idx] = {"start": p.get("start"),
                                          "end": p.get("end"),
                                          "reason": p.get("reason", "")}
    elif ev["kind"] == "model_verdict":
        st.verdicts = [v for v in st.verdicts
                       if v.get("fit_id") != p.get("fit_id")]
        st.verdicts.append({"fit_id": p.get("fit_id"),
                            "model": p.get("model"),
                            "verdict": p.get("verdict"),
                            "reason": p.get("reason", ""),
                            "event_id": ev["id"]})
        # a conflicting human decision: same model accepted and rejected
        verdicts_by_model = {}
        for v in st.verdicts:
            verdicts_by_model.setdefault(v["model"], []).append(v["verdict"])
        for model, vs in verdicts_by_model.items():
            if "accept" in vs and "reject" in vs:
                st.conflicts = [c for c in st.conflicts
                                if c.get("model") != model]
                st.conflicts.append({"model": model, "status": "conflict",
                                     "event_id": ev["id"],
                                     "verdicts": list(set(vs))})
    elif ev["kind"] == "decision_conflict":
        st.conflicts.append({"model": p.get("model"),
                             "status": "unresolved_conflict",
                             "event_id": ev["id"], "payload": p})
    elif ev["kind"] == "review_note":
        pass
    # derived recompute (deterministic; historical artifacts kept by id)
    _rederive(ev, st, store, artifacts, budget=budget)


def _rederive(ev: dict, st: BranchState, store: MaterialStore,
              artifacts: dict, *, budget: int) -> None:
    needs_integration = ev["kind"] in (
        "material_committed", "first_injection_discard",
        "blank_rule_set", "integration_endpoint")
    needs_fit = needs_integration or ev["kind"] == "model_verdict"
    if needs_integration and st.stream_versions.get("power") and \
            st.stream_versions.get("injections"):
        art = compute_integrations(st, store)
        art["branch_id"] = st.branch_id
        st.integration_art_id = art.get("id")
        artifacts[art["id"]] = art
    if needs_fit and st.integration_art_id and \
            st.stream_versions.get("concentrations") and \
            st.stream_versions.get("cell_volume"):
        integ_art = artifacts[st.integration_art_id]
        fit = compute_fits(st, store, integ_art, budget=budget)
        fit["branch_id"] = st.branch_id
        st.fits = [fit] if fit.get("id") else []
        if fit.get("id"):
            artifacts[fit["id"]] = fit


def _checkpoint(ev: dict, st: BranchState, artifacts: dict) -> dict[str, Any]:
    return {"event_id": ev["id"], "seq": ev["seq"], "kind": ev["kind"],
            "branch_id": st.branch_id,
            "stream_heads": dict(st.stream_heads),
            "integration_id": st.integration_art_id,
            "fit_ids": [f.get("id") for f in st.fits],
            "discarded": sorted(st.discarded),
            "blank_rule": st.blank_rule,
            "conflicts": list(st.conflicts)}


def staleness(art: dict, state: BranchState, store: MaterialStore) -> bool:
    """True when a newer material head makes this derived artifact stale."""
    parents = art.get("parents", {})
    for kind, ref in parents.items():
        if kind in ("blank_rule", "rule_version") or not ref:
            continue
        head = state.stream_versions.get(kind)
        if head and head != ref.get("id"):
            # concentration revision makes MODELS stale but not integrations
            if kind == "concentrations" and art["id"].startswith("intg_"):
                continue
            return True
    if art["id"].startswith("fit_"):
        integ_ref = parents.get("integration")
        if integ_ref and integ_ref != state.integration_art_id:
            return True
        # explicit rule: concentration revision expires models
        conc_head = state.stream_versions.get("concentrations")
        conc_ref = parents.get("concentrations")
        if conc_head and conc_ref and conc_head != conc_ref.get("id"):
            return True
    return False
