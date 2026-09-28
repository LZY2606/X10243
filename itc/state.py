"""从只追加事件重建分支状态（支持时间旅行：重建到任意 seq）。

每个分支是线性的（建分支时复制祖先事件），所以重建就是按 seq fold。
派生结果记录 parent_refs；材料/规则被修订后，旧结果不会被覆盖，
而是被标记 superseded/stale，连同当时结论一起保留。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def rebuild_state(store, project_id: str, branch: str, until_seq: Optional[int] = None) -> Dict[str, Any]:
    events = store.events_until(project_id, branch, until_seq) if until_seq is not None \
        else store.events(project_id, branch)

    materials: Dict[str, Dict[str, Any]] = {}
    rules: Dict[str, Dict[str, Any]] = {}
    integrations: Dict[str, Dict[str, Any]] = {}   # slot(role) -> 当前结果
    subtractions: Dict[str, Any] = {}
    fits: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    branch_points: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for e in events:
        kind, p = e["kind"], e["payload"]
        if kind == "material_imported":
            key = f"{p['kind']}|{p['role']}"
            materials[key] = {
                "material_id": p["material_id"], "kind": p["kind"], "role": p["role"],
                "label": p.get("label", p["source_name"]), "ordinal": p["ordinal"],
                "source_name": p["source_name"], "event_seq": e["seq"], "event_id": e["event_id"],
                "columns": p.get("columns", {}), "scalars": p.get("scalars", {}),
                "meta": p.get("meta", {}), "diagnostics": p.get("diagnostics", []),
            }
        elif kind == "rules_updated":
            rules[p["ruleset"]] = {"ruleset": p["ruleset"], "ordinal": p["ordinal"],
                                   "rules": p["rules"], "change_note": p.get("change_note", ""),
                                   "event_seq": e["seq"], "event_id": e["event_id"]}
        elif kind == "integration_completed":
            integrations[p["slot"]] = {"event_seq": e["seq"], "event_id": e["event_id"], **p}
        elif kind == "blank_subtracted":
            subtractions[p["slot"]] = {"event_seq": e["seq"], "event_id": e["event_id"], **p}
        elif kind == "fit_completed":
            fits.append({"event_seq": e["seq"], "event_id": e["event_id"], **p})
        elif kind == "decision":
            decisions.append({"event_seq": e["seq"], "event_id": e["event_id"], **p})
        elif kind == "branch_created":
            branch_points.append({"event_seq": e["seq"], **p})

    current_mat_ids = {k: m["material_id"] for k, m in materials.items()}
    current_rule_ord = {k: r["ordinal"] for k, r in rules.items()}

    # 派生结果过期/被取代标记（不改写结果本身，只在重建视图标注）
    latest_fit_slot: Dict[str, int] = {}
    for i, f in enumerate(fits):
        latest_fit_slot[f.get("slot", "default")] = i
    for i, f in enumerate(fits):
        f["superseded"] = latest_fit_slot.get(f.get("slot", "default")) != i
        f["stale"] = _parents_stale(f.get("parents", {}), current_mat_ids, current_rule_ord)
        f["decision_state"] = _decision_state(f["result_id"], decisions)
    for slot, integ in integrations.items():
        integ["stale"] = _parents_stale(integ.get("parents", {}), current_mat_ids, current_rule_ord)

    active_diag: List[Dict[str, Any]] = []
    for m in materials.values():
        for d in m.get("diagnostics", []):
            active_diag.append({**d, "where": f"material:{m['kind']}/{m['role']}#v{m['ordinal']+1}"})

    return {
        "branch": branch,
        "head_seq": events[-1]["seq"] if events else 0,
        "materials": materials,
        "rules": rules,
        "integrations": integrations,
        "subtractions": subtractions,
        "fits": fits,
        "decisions": decisions,
        "branch_points": branch_points,
        "events": events,
        "diagnostics": active_diag,
    }


def _parents_stale(parents: Dict[str, Any], current_mat_ids: Dict[str, str],
                   current_rule_ord: Dict[str, int]) -> bool:
    for ref in parents.get("materials", []):
        cur = current_mat_ids.get(f"{ref['kind']}|{ref['role']}")
        if cur is not None and cur != ref["material_id"]:
            return True
    for ref in parents.get("rules", []):
        cur = current_rule_ord.get(ref["ruleset"])
        if cur is not None and cur != ref["ordinal"]:
            return True
    return False


def _decision_state(result_id: str, decisions: List[Dict[str, Any]]) -> Optional[str]:
    state = None
    for d in decisions:
        if d.get("target_id") == result_id:
            state = d.get("action")
    return state
