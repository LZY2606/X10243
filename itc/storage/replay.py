"""事件重放：在任意事件点恢复材料版本、决定与派生 DAG。

分支分叉后，每个分支只见到自己 fork 点之后追加的事件；
分支流通过 fork_event_id 继承主干前缀事件（物化到分支视图，不复制材料）。
"""
from __future__ import annotations

from typing import Any

from ..core.constants import EVT_MATERIAL_UPLOADED
from ..core.engine import recompute
from ..core.fitter import Budget


def branch_event_stream(store, project_id: str, branch_id: str,
                        upto: str | None = None) -> list[dict[str, Any]]:
    branch = store.conn.execute(
        "SELECT * FROM branches WHERE id=?", (branch_id,)).fetchone()
    main = store.get_main_branch(project_id)
    events = store.events(project_id, main["id"], upto=None)
    if branch["id"] != main["id"]:
        fork = branch["fork_event_id"]
        cut = next((i for i, e in enumerate(events) if e["id"] == fork),
                   len(events) - 1)
        prefix = events[:cut + 1]
        suffix = store.events(project_id, branch["id"])
        events = prefix + suffix
    if upto:
        cut = next((i for i, e in enumerate(events) if e["id"] == upto),
                   None)
        if cut is not None:
            events = events[:cut + 1]
    return events


def state_at(store, project_id: str, branch_id: str,
             upto: str | None = None, budget: Budget | None = None,
             ) -> dict[str, Any]:
    events = branch_event_stream(store, project_id, branch_id, upto)
    materials: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    material_timeline: dict[str, list[dict[str, Any]]] = {}

    for e in events:
        p = e["payload"]
        if e["type"] == EVT_MATERIAL_UPLOADED:
            row = store.conn.execute(
                "SELECT * FROM materials WHERE id=? AND version=?",
                (p["material_id"], p["version"])).fetchone()
            if row is None:
                continue
            import json
            payload = json.loads(row["payload_json"])
            materials[p["slot"]] = {
                "material_id": p["material_id"],
                "version": p["version"],
                "raw_digest": p["raw_digest"],
                "payload": payload,
                "event_id": e["id"],
            }
            material_timeline.setdefault(p["slot"], []).append(
                {"material_id": p["material_id"], "version": p["version"],
                 "event_id": e["id"], "raw_digest": p["raw_digest"],
                 "label": p.get("label")})
        elif e["type"] == "decision_applied":
            decisions.append({"key": p["key"], "value": p["value"],
                              "kind": p["kind"], "event_id": e["id"],
                              "seq": e["seq"]})
        elif e["type"] == "review_verdict":
            reviews.append({"event_id": e["id"], **p})

    result = recompute(materials, decisions, budget=budget)
    result.update({
        "head_event_id": events[-1]["id"] if events else None,
        "events": events,
        "materials": materials,
        "material_timeline": material_timeline,
        "decisions": decisions,
        "reviews": reviews,
        "missing_slots": [s for s in
                          ("power", "injections", "concentrations",
                           "temperature", "cell_volume", "baseline",
                           "blank") if s not in materials],
    })
    return result
