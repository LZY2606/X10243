"""Replay events into per-branch state; immutable material / mutable decision layering.

Two-pass replay:
  pass 1 (created_order): project_created builds an empty state; branch_forked
    clones the source branch state at the fork's at_seq.
  pass 2 (created_order): all material/link/decision events mutate the branch
    they belong to, so each branch carries its own material versions and
    decision history while sharing an event chain / blob store.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

from .parsing import (parse_baseline, parse_concentrations, parse_conditions,
                      parse_power, parse_schedule)
from .util import digest

PARSERS = {
    "power": parse_power,
    "schedule": parse_schedule,
    "concentrations": parse_concentrations,
    "conditions": parse_conditions,
    "baseline": parse_baseline,
}
MATERIAL_KINDS = tuple(PARSERS)


def material_identity(parsed: dict) -> str:
    basis = {"kind": parsed["kind"], "metadata": parsed["metadata"]}
    if "samples" in parsed:
        basis["points"] = [[s["index"], s["raw_time"], s["raw_power"]]
                           for s in parsed["samples"]]
        basis["units"] = parsed.get("units")
    if "injections" in parsed:
        basis["points"] = [[j["ordinal"], j["raw_time"], j["raw_volume"]]
                           for j in parsed["injections"]]
        basis["units"] = parsed.get("units")
    if "entries" in parsed:
        basis["entries"] = {k: v.get("raw") for k, v in parsed["entries"].items()}
    if "anchors" in parsed:
        basis["anchors"] = [[a["raw_time"], a["raw_level"]] for a in parsed["anchors"]]
        basis["method"] = parsed.get("method")
    return digest(basis, length=18, prefix=f"mat_{parsed['kind']}_")


def _new_state(project_id, title):
    return {
        "project_id": project_id, "title": title,
        "series": {}, "series_order": [],
        "blank_links": {},
        "decisions": {},
        "forks": [],
        "diagnostics": [],
        "notes": [],
    }


def _sorted(events):
    return sorted(events, key=lambda e: e["created_order"])


def replay(events: list[dict], blob_lookup: Callable[[str], Optional[str]]):
    states: dict[str, dict] = {}
    heads: dict[str, str] = {}
    ordered = _sorted(events)

    def parse_material(kind: str, blob_id: str):
        content = blob_lookup(blob_id)
        if content is None:
            raise KeyError(f"blob {blob_id} missing during replay")
        return PARSERS[kind](content, path=blob_id)

    # Pass 1: branch skeletons + fork cloning at at_seq.
    for ev in ordered:
        t, p = ev["type"], ev["payload"]
        if t == "project_created":
            states[ev["branch"]] = _new_state(p["project_id"], p.get("title", p["project_id"]))
        elif t == "branch_forked":
            src_branch = p["from_branch"]
            at_seq = p["at_seq"]
            cloned = _replay_branch(events, src_branch, at_seq, blob_lookup,
                                    existing=states.get(src_branch))
            cloned["forks"].append({"branch": p["new_branch"], "from": src_branch,
                                    "at_seq": at_seq, "reason": p.get("reason", "")})
            states[p["new_branch"]] = cloned

    # Pass 2: apply each event to its branch state.
    for ev in ordered:
        t = ev["type"]
        if t in ("project_created", "branch_forked"):
            heads[ev["branch"]] = ev["hash"]
            continue
        if ev["branch"] not in states:
            # Event predates a project_created we saw (shouldn't happen); skip.
            continue
        _apply(states[ev["branch"]], ev, parse_material)
        heads[ev["branch"]] = ev["hash"]

    return states, heads


def _replay_branch(events, branch, at_seq, blob_lookup, *, existing=None):
    sub = sorted((e for e in events if e["branch"] == branch and e["seq"] <= at_seq),
                 key=lambda e: e["seq"])
    state = None
    # reuse already-reconstructed state if it contains only events <= at_seq
    if existing is not None:
        state = copy.deepcopy(existing)
        # rollback decisions/materials newer than at_seq is complex; build fresh.
        state = None
    state = _new_state("", "")
    for ev in sub:
        if ev["type"] == "project_created":
            state = _new_state(ev["payload"]["project_id"],
                               ev["payload"].get("title", ev["payload"]["project_id"]))
            continue
        if state is None:
            continue
        _apply(state, ev,
               lambda kind, bid: PARSERS[kind](blob_lookup(bid), path=bid))
    return state


def _apply(state, ev, parse_material):
    p = ev["payload"]
    t = ev["type"]
    if t == "series_added":
        state["series"][p["series_id"]] = {
            "id": p["series_id"], "role": p["role"], "label": p.get("label", ""),
            "materials": {}, "material_versions": {}}
        if p["series_id"] not in state["series_order"]:
            state["series_order"].append(p["series_id"])
        state["decisions"].setdefault(p["series_id"], [])
    elif t == "material_added":
        sid = p["series_id"]
        s = state["series"][sid]
        parsed = parse_material(p["kind"], p["blob_id"])
        mid = material_identity(parsed)
        versions = s["material_versions"].setdefault(p["kind"], [])
        version_no = len(versions) + 1
        record = {"id": mid, "kind": p["kind"], "version": version_no,
                  "blob_id": p["blob_id"], "path": p.get("path"),
                  "event_id": ev["id"], "seq": ev["seq"], "parsed": parsed}
        versions.append(record)
        s["materials"][p["kind"]] = record
        state["diagnostics"] = [d for d in state["diagnostics"]
                                if not (d.get("series_id") == sid and d.get("kind") == p["kind"]
                                        and d.get("version", 10**9) <= version_no)]
        for d in parsed.get("diagnostics", []):
            state["diagnostics"].append({**d, "series_id": sid, "kind": p["kind"],
                                         "material_id": mid, "version": version_no})
    elif t == "blank_linked":
        state["blank_links"][p["sample_series_id"]] = {
            "blank_series_id": p["blank_series_id"],
            "rule": p.get("rule", {"mode": "ordinal"}),
            "event_id": ev["id"], "seq": ev["seq"]}
    elif t == "decision_added":
        d = dict(p["decision"])
        d["event_id"] = ev["id"]
        d["seq"] = ev["seq"]
        state["decisions"].setdefault(p["series_id"], []).append(d)
    elif t == "note":
        state["notes"].append({"text": p.get("text"), "seq": ev["seq"]})


def all_material_digests(state, sid) -> dict:
    return {kind: rec["id"] for kind, rec in state["series"][sid]["materials"].items()}
