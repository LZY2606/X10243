"""Application workflow layer: import, revise, decide, persist derived results.

Each call appends machine-readable events and stores computation results with
``input_summary``, ``rule_version`` and ``parent_versions``.
"""

from __future__ import annotations

import json

from . import engine, storage
from .errors import IncompleteInput
from .models import (parse_concentrations, parse_conditions, parse_injections,
                     parse_series)

PARSE_FOR = {
    "power": lambda txt, fn: parse_series(txt, "power", "main", fn),
    "blank_power": lambda txt, fn: parse_series(txt, "blank_power", "blank", fn),
    "baseline": lambda txt, fn: parse_series(txt, "baseline", "main", fn),
    "injections": lambda txt, fn: parse_injections(txt, "injections", "main", fn),
    "blank_injections":
        lambda txt, fn: parse_injections(txt, "blank_injections", "blank", fn),
    "concentrations": lambda txt, fn: parse_concentrations(txt, fn),
    "conditions": lambda txt, fn: parse_conditions(txt, fn),
}

ROLE_FOR = {"blank_power": "blank", "blank_injections": "blank"}


def _store_material(conn, project_id, kind, raw_text, filename,
                    supersedes=None):
    parsed = PARSE_FOR[kind](raw_text, filename)
    role = ROLE_FOR.get(kind, "main")
    return storage.insert_material(
        conn, project_id, kind, parsed.to_dict(), raw_text, role=role,
        source_filename=filename, supersedes=supersedes)


def import_bundle(conn, project_id, files: dict[str, tuple[str, str]],
                  etype="initial_import", event_note=None,
                  created_at=None) -> str:
    """files: {kind: (raw_text, filename)}.  One transaction per event."""
    refs = []
    superseded = []
    for kind, (raw_text, filename) in files.items():
        prev = storage.latest_material(
            conn, project_id, kind, ROLE_FOR.get(kind, "main"))
        mid = _store_material(conn, project_id, kind, raw_text, filename,
                              supersedes=prev["material_id"] if prev else None)
        refs.append({"material_id": mid, "kind": kind,
                     "version": int(mid.split("_v")[-1]),
                     "source_filename": filename})
        if prev:
            superseded.append(prev["material_id"])
    payload = {"materials": refs}
    if event_note:
        payload["note"] = event_note
    if superseded:
        payload["supersedes"] = superseded
    branch = storage.main_branch(conn, project_id)
    eid = storage.insert_event(conn, project_id, branch["branch_id"], etype,
                               payload, created_at=created_at)
    return eid


def revise_material(conn, project_id, kind, raw_text, filename, note,
                    created_at=None) -> dict:
    """Insert a new material version, append event, invalidate stale derivations.

    Concentration revisions invalidate binding fits only — raw per-injection
    integrals are independent of concentrations and stay valid.
    """
    eid = import_bundle(conn, project_id, {kind: (raw_text, filename)},
                        etype="material_revision", event_note=note,
                        created_at=created_at)
    if kind == "concentrations":
        storage.mark_stale_results(conn, project_id, ["fit"])
    elif kind in ("power", "baseline", "injections", "blank_power",
                  "blank_injections"):
        storage.mark_stale_results(conn, project_id, ["integration", "fit"])
    return {"event_id": eid}


def record_decision(conn, project_id, name, value, adopted=True,
                    reason="", branch_id=None, extra=None,
                    created_at=None) -> dict:
    branch = storage.get_branch(conn, branch_id) if branch_id else \
        storage.main_branch(conn, project_id)
    branch_id = branch["branch_id"]
    events = storage.branch_events(conn, branch_id)
    parent = events[-1]["event_id"] if events else None
    decision = {"name": name, "value": value, "reason": reason}
    if extra:
        decision.update(extra)
    eid = storage.insert_event(
        conn, project_id, branch_id, "decision", {"decision": decision},
        parent_event_id=parent, adopted=adopted, created_at=created_at)

    view = engine.derive_view(conn, project_id, branch_id, eid)
    parent_versions = {f"{k[0]}:{k[1]}": v["material_id"]
                       for k, v in view.materials.items()}
    if adopted and name in ("discard_injections", "blank_correspondence_rule",
                            "manual_adjustment"):
        storage.insert_result(
            conn, project_id, "integration", branch_id, eid,
            _integration_summary(view), parent_versions,
            {"injections": view.integrations},
            extra=name)
    if adopted and name == "accept_model":
        fit = view.fit
        storage.insert_result(
            conn, project_id, "fit", branch_id, eid,
            {"n_points": len(fit["usable_indices"]) if fit else 0},
            {**(fit.get("input_materials", {}) if fit else {})},
            fit if fit else {}, status=("ok" if fit else "incomplete"),
            extra=json.dumps(value, sort_keys=True))
    if not adopted:
        # keep the rejected outcome too, with explicit provenance
        storage.insert_result(
            conn, project_id, "rejected_decision", branch_id, eid,
            {"decision": decision}, parent_versions, {"rejected": True},
            status="rejected", extra=name)
    return {"event_id": eid, "branch_id": branch_id,
            "view": view.to_dict()}


def _integration_summary(view):
    ok = [i for i in view.integrations if i["q_ucal"] is not None]
    return {
        "n_injections": len(view.integrations),
        "n_integrated": len(ok),
        "overlapping": [i["index"] for i in view.integrations
                        if i["overlap_with_next"]],
        "gap_rejected": [i["index"] for i in view.integrations
                         if i["status"] == "rejected_gap"],
        "open_endpoint": [i["index"] for i in view.integrations
                          if i["endpoint_open"]],
        "unsubtracted": [i["index"] for i in view.integrations
                         if i["dilution_status"] != "subtracted"],
        "discarded": [i["index"] for i in view.integrations if i["discarded"]],
    }


def persist_derived(conn, project_id, branch_id=None):
    branch = storage.get_branch(conn, branch_id) if branch_id else \
        storage.main_branch(conn, project_id)
    head = storage.branch_events(conn, branch["branch_id"])[-1]
    view = engine.derive_view(conn, project_id, branch["branch_id"],
                              head["event_id"])
    parent_versions = {f"{k[0]}:{k[1]}": v["material_id"]
                       for k, v in view.materials.items()}
    ids = []
    if view.integrations:
        ids.append(storage.insert_result(
            conn, project_id, "integration", branch["branch_id"],
            head["event_id"], _integration_summary(view), parent_versions,
            {"injections": view.integrations}))
    if view.fit:
        ids.append(storage.insert_result(
            conn, project_id, "fit", branch["branch_id"], head["event_id"],
            {"n_points": len(view.fit["usable_indices"])},
            view.fit.get("input_materials", {}), view.fit))
    return ids


def create_decision_branch(conn, project_id, name, fork_event_id,
                           created_at=None):
    return storage.create_branch(conn, project_id, name, fork_event_id,
                                 created_at=created_at)
