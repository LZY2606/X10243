"""Event-sourced replay engine.

Layering:

* immutable layer - materials (six kinds versioned separately) and the
  content-addressed integration/fit results derived from them;
* mutable layer - decisions and rule versions expressed as events.

Every computation records an input summary, rule/parameter versions and
parent dependencies inside its content hash, so equivalent inputs always get
the same identity and order. Checking out an old event replays only the
ancestor chain and restores the conclusions that were visible then; newer
computations never overwrite history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .errors import ItcError
from .ids import content_id
from .integrate import integrate_injections
from .models import fit_models

DEFAULT_WINDOW_RULE = {"pre_s": 5.0, "post_s": 60.0}
DEFAULT_BUDGET = 4000


def new_project(project_id: str = "proj_demo") -> dict[str, Any]:
    return {
        "id": project_id,
        "schema": "itc-project/v1",
        "seq": 0,
        "materials": {},
        "results": {},
        "events": [],
        "branches": {},
    }


def _head_key(kind: str, role: str) -> str:
    return f"{kind}:{role}"


@dataclass
class RuleVersion:
    version: dict[str, Any]
    version_id: str


def _rule_version(kind: str, rule: dict[str, Any]) -> RuleVersion:
    version = {"kind": kind, **rule}
    return RuleVersion(version, content_id("rule", version))


def _blank_result_recompute(state: dict[str, Any], project: dict[str, Any], event_id: str) -> None:
    heads = state["heads"]
    need = ("power:main", "injections:main", "baseline:main")
    if any(k not in heads for k in need):
        return
    power = project["materials"][heads["power:main"]]
    injections = project["materials"][heads["injections:main"]]
    baseline = project["materials"][heads["baseline:main"]]
    blank = blank_power = blank_inj = None
    if heads.get("blank:main"):
        blank = project["materials"][heads["blank:main"]]
        blank_power = project["materials"][blank["payload"]["power_material_id"]]
        blank_inj = project["materials"][blank["payload"]["injections_material_id"]]
    integration = integrate_injections(
        power,
        injections,
        baseline,
        window_rule=state["window_rule"],
        blank=blank,
        blank_power=blank_power,
        blank_injections=blank_inj,
        blank_rule=state["blank_rule"],
        first_injection_discarded=state["drop_first"],
    )
    _store_result(project, integration, event_id)
    state["current_integration_id"] = integration["id"]

    conc_id = heads.get("concentration:main")
    cv_id = heads.get("cell_volume:main")
    state["current_fit_id"] = None
    if conc_id and cv_id:
        concentration = project["materials"][conc_id]
        cell_volume = project["materials"][cv_id]
        fit = fit_models(
            integration, concentration, cell_volume, budget=state.get("fit_budget", DEFAULT_BUDGET)
        )
        _store_result(project, fit, event_id)
        state["current_fit_id"] = fit["id"]


def _store_result(project: dict[str, Any], result: dict[str, Any], event_id: str) -> None:
    record = project["results"].setdefault(
        result["id"], {"result": result, "event_ids": []}
    )
    if event_id not in record["event_ids"]:
        record["event_ids"].append(event_id)


def _new_state() -> dict[str, Any]:
    wv = _rule_version("window", DEFAULT_WINDOW_RULE)
    return {
        "heads": {},
        "window_rule": wv.version,
        "window_rule_id": wv.version_id,
        "blank_rule": None,
        "blank_rule_id": None,
        "drop_first": False,
        "fit_budget": DEFAULT_BUDGET,
        "current_integration_id": None,
        "current_fit_id": None,
        "decision_event_ids": [],
    }


def apply_event(project: dict[str, Any], event: dict[str, Any]) -> None:
    """Apply one event. The event must already carry stable id/order fields."""
    branch = event["branch"]
    if branch == "main":
        state = project["branches"].setdefault("main", {"state": _new_state(), "spec": {"name": "main"}})["state"]
    else:
        if branch not in project["branches"]:
            raise KeyError(f"branch {branch} not created before its first event")
        state = project["branches"][branch]["state"]

    etype = event["type"]
    payload = event.get("payload", {})
    if etype == "IMPORT_MATERIAL":
        material = payload["material"]
        project["materials"][material["id"]] = material
        if payload.get("head", True):
            key = _head_key(material["kind"], payload.get("role", material["role"]))
            state["heads"][key] = material["id"]
    elif etype == "SUPERSEDE_HEAD":
        key = _head_key(payload["kind"], payload["role"])
        if payload["material_id"] not in project["materials"]:
            raise KeyError("cannot supersede with unknown material")
        state["heads"][key] = payload["material_id"]
    elif etype == "SET_WINDOW_RULE":
        rv = _rule_version("window", {"pre_s": float(payload["pre_s"]), "post_s": float(payload["post_s"])})
        state["window_rule"] = rv.version
        state["window_rule_id"] = rv.version_id
    elif etype == "SET_BLANK_RULE":
        rule = {"correspondence": payload["correspondence"], "scale": float(payload.get("scale", 1.0))}
        rv = _rule_version("blank", rule)
        state["blank_rule"] = rv.version
        state["blank_rule_id"] = rv.version_id
    elif etype == "DROP_FIRST":
        state["drop_first"] = bool(payload["discard"])
    elif etype == "SET_FIT_BUDGET":
        state["fit_budget"] = int(payload["budget"])
    elif etype == "MODEL_DECISION":
        state["decision_event_ids"].append(event["id"])
    elif etype == "FORK_BRANCH":
        child = payload["name"]
        project["branches"][child] = {
            "spec": {"name": child, "forked_from": branch, "fork_event": event["id"]},
            "state": copy.deepcopy(state),
        }
    else:
        raise KeyError(f"unknown event type {etype}")

    project["events"].append(event)
    if etype not in ("FORK_BRANCH", "MODEL_DECISION"):
        try:
            _blank_result_recompute(state, project, event["id"])
            state["compute_error"] = None
        except ItcError as exc:
            state["compute_error"] = exc.to_dict()


def make_event(
    project: dict[str, Any],
    branch: str,
    etype: str,
    payload: dict[str, Any],
    *,
    actor: str = "reviewer",
    note: str = "",
) -> dict[str, Any]:
    branch_seq = sum(1 for e in project["events"] if e["branch"] == branch) + 1
    parent_id = None
    prior = [e for e in project["events"] if e["branch"] == branch]
    if prior:
        parent_id = prior[-1]["id"]
    project["seq"] += 1
    body = {
        "project": project["id"],
        "branch": branch,
        "branch_seq": branch_seq,
        "parent_event_id": parent_id,
        "type": etype,
        "payload": payload,
        "actor": actor,
        "note": note,
        "seq": project["seq"],
    }
    event = dict(body)
    event["id"] = content_id("evt", body)
    return event


def commit_event(project: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    apply_event(project, event)
    return event


def ancestor_events(project: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    """Events visible on a branch: parent chain up to the fork, then its own."""
    chain: list[dict[str, Any]] = []
    cur = branch
    seen: set[str] = set()
    while cur not in seen:
        seen.add(cur)
        own = [e for e in project["events"] if e["branch"] == cur]
        if cur == branch:
            chain = own + chain
        else:
            fork = project["branches"][cur]["spec"].get("fork_event")
            upto = own
            if fork:
                upto = []
                for e in own:
                    upto.append(e)
                    if e["id"] == fork:
                        break
            chain = upto + chain
        spec = project["branches"].get(cur, {}).get("spec", {})
        cur = spec.get("forked_from")
        if not cur:
            break
    chain.sort(key=lambda e: (e["seq"], e["branch"]))
    return chain


def checkout(project: dict[str, Any], branch: str, event_id: str | None = None) -> dict[str, Any]:
    """Replay only the ancestor chain up to ``event_id`` (history-safe)."""
    fork_spec = project["branches"].get(branch, {}).get("spec", {})
    target_project = {
        "id": project["id"],
        "schema": project["schema"],
        "seq": 0,
        "materials": {},
        "results": {},
        "events": [],
        "branches": {},
    }
    chain = ancestor_events(project, branch)
    applied: list[dict[str, Any]] = []
    for event in chain:
        apply_event(target_project, copy.deepcopy(event))
        applied.append(event)
        if event_id is not None and event["id"] == event_id:
            break
    state = target_project["branches"].get(branch, target_project["branches"].get("main"))["state"]
    return {
        "project": target_project,
        "state": state,
        "events": applied,
        "spec": fork_spec,
    }
