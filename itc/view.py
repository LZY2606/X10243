"""Projected state -> stable, machine-readable review view.

Every displayed entity carries a stable identity (material version event id,
integration/fit event id, per-shot derived id). Staleness is *computed* from
parent dependencies versus current material heads; stale results remain on
record and are never silently replaced.
"""

from __future__ import annotations

from .projector import State, replay

# material kinds whose revision invalidates a fitted binding model
FIT_INPUT_KINDS = ("concentration", "temperature", "cell_volume", "schedule")
# material kinds whose revision invalidates raw per-shot integration
INTEGRAL_INPUT_KINDS = ("power", "baseline", "schedule")


def build_view(events: list[dict], *, project_id: str, branch_id: str,
               branch_name: str, upto: str | None = None) -> dict:
    replayed = replay(events, project_id=project_id)
    material_heads = _material_heads(events)
    decisions = _current_decisions(replayed)
    integrations = []
    for rec in replayed.integrations:
        integrations.append(_integration_view(rec, material_heads))
    current_integration = next(
        (i for i in reversed(integrations) if i["status"] == "current"), None)
    fits = []
    for rec in replayed.fits:
        fits.append(_fit_view(rec, material_heads, integrations))
    current_fit = next((f for f in reversed(fits) if f["status"] == "current"), None)
    return {
        "project_id": project_id,
        "branch_id": branch_id,
        "branch_name": branch_name,
        "upto": upto,
        "head_seq": events[-1]["seq"] if events else 0,
        "name": replayed.name,
        "materials": _materials_view(events),
        "decisions": decisions,
        "integrations": integrations,
        "fits": fits,
        "current_integration": current_integration,
        "current_fit": current_fit,
        "diagnostics": replayed.diagnostics,
        "timeline": _timeline(events),
    }


def _material_heads(events: list[dict]) -> dict[str, str]:
    heads: dict[str, str] = {}
    for ev in events:
        if ev["type"] == "material_edit":
            kind = ev["payload"]["material"]["kind"]
            heads[kind] = ev["event_id"]
    return heads


def _materials_view(events: list[dict]) -> dict:
    out: dict[str, list] = {}
    for ev in events:
        if ev["type"] != "material_edit":
            continue
        m = ev["payload"]["material"]
        out.setdefault(m["kind"], []).append({
            "id": ev["event_id"],
            "seq": ev["seq"],
            "kind": m["kind"],
            "version": m["version"],
            "origin": m["origin"],
            "unit": m.get("unit"),
            "raw": m.get("raw"),
            "quantity": m.get("quantity"),
            "extra": m.get("extra", {}),
            "n_points": len(m["series"]["points"]) if m.get("series") else None,
            "n_injections": len(m.get("injections", [])),
        })
    return out


def _current_decisions(state: State) -> dict:
    current: dict = {}
    for d in state.decisions:
        p = d["payload"]
        if d["type"] == "discard_shots":
            current["discard_shots"] = d
        elif d["type"] == "blank_rule":
            current["blank_rule"] = d
        elif d["type"] == "integration_endpoints":
            current.setdefault("endpoints", {})[p.get("shot_index")] = d
        elif d["type"] == "parameter_override":
            current.setdefault("overrides", {})[p.get("name")] = d
        elif d["type"] == "manual_decision":
            current.setdefault("manual", []).append(d)
        elif d["type"] == "review_note":
            current.setdefault("notes", []).append(d)
    return current


def _staleness(parents: dict, material_heads: dict[str, str]) -> dict:
    """Compare recorded parent event ids to current material heads.

    Returns status + every specific reason, so the UI can say which revision
    invalidated a derived result.
    """
    reasons = []
    for kind, parent_id in parents.items():
        head = material_heads.get(kind)
        if head is None:
            reasons.append({"code": "missing_input", "kind": kind})
        elif head != parent_id:
            reasons.append({"code": "material_revised", "kind": kind,
                            "used": parent_id, "current": head})
    return {"status": "stale" if reasons else "current", "reasons": reasons}


def _integration_view(rec, material_heads) -> dict:
    stale = _staleness(rec.parents, material_heads)
    # endpoint/discard/blank-rule decisions are parented by id as well
    decision_parents = rec.settings.get("decision_parents", {})
    shots = []
    for shot in rec.shots:
        s = dict(shot)
        s["id"] = f"shot-{rec.event_id}-{shot['index']}"
        shots.append(s)
    return {
        "id": rec.event_id,
        "seq": rec.seq,
        "status": stale["status"],
        "stale_reasons": stale["reasons"],
        "settings": rec.settings,
        "input_summary": rec.input_summary,
        "parents": rec.parents,
        "decision_parents": decision_parents,
        "shots": shots,
        "diagnostics": rec.diagnostics,
        "ts": rec.ts,
    }


def _fit_view(rec, material_heads, integration_views) -> dict:
    reasons = _staleness(rec.parents, material_heads)["reasons"]
    referenced = next((i for i in integration_views if i["id"] == rec.integration_id), None)
    if referenced is None:
        reasons.append({"code": "integration_missing", "used": rec.integration_id})
    elif referenced["status"] != "current":
        reasons.append({"code": "integration_stale", "used": rec.integration_id,
                        "reasons": referenced["stale_reasons"]})
    # discarded shots/ blank rules also invalidate a fit through the
    # integration settings; compare decision parents if present.
    return {
        "id": rec.event_id,
        "seq": rec.seq,
        "integration_id": rec.integration_id,
        "status": "stale" if reasons else "current",
        "stale_reasons": reasons,
        "settings": rec.settings,
        "input_summary": rec.input_summary,
        "parents": rec.parents,
        "candidates": rec.candidates,
        "equivalent": [c for c in rec.candidates if c.get("equivalent_best")],
        "best": next((c for c in rec.candidates if c.get("rank") == 1), None),
        "diagnostics": rec.diagnostics,
        "ts": rec.ts,
    }


def _timeline(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
        rows.append({
            "event_id": ev["event_id"],
            "seq": ev["seq"],
            "branch_id": ev["branch_id"],
            "type": ev["type"],
            "ts": ev["ts"],
            "label": _label(ev),
            "payload_refs": _refs(ev),
        })
    return rows


def _label(ev: dict) -> str:
    p = ev["payload"]
    if ev["type"] == "project_created":
        return f"项目创建：{p.get('name', '')}"
    if ev["type"] == "material_edit":
        return f"材料版本 {p['material']['kind']} v{p['material']['version']}（{p['material']['origin']}）"
    if ev["type"] == "integrated":
        return f"注射积分（{len(p['shots'])} 针）"
    if ev["type"] == "fitted":
        return f"绑定模型候选 {len(p['candidates'])} 个"
    if ev["type"] == "discard_shots":
        return f"首注舍弃：{p.get('shots')}"
    if ev["type"] == "blank_rule":
        return f"稀释热对应规则：{p.get('rule')}"
    if ev["type"] == "manual_decision":
        return f"人工决定：{p.get('decision')}（{p.get('outcome')}）"
    if ev["type"] == "parameter_override":
        return f"参数调整：{p.get('name')}={p.get('value')}"
    if ev["type"] == "integration_endpoints":
        return f"积分端点：第 {p.get('shot_index')} 针"
    if ev["type"] == "review_note":
        return f"审阅备注：{p.get('text', '')[:40]}"
    return ev["type"]


def _refs(ev: dict) -> dict:
    p = ev["payload"]
    refs = {}
    if "parents" in p:
        refs["parents"] = p["parents"]
    if "integration_id" in p:
        refs["integration_id"] = p["integration_id"]
    if ev["type"] == "material_edit":
        refs["material"] = ev["event_id"]
    if "basis" in p:
        refs["basis"] = p["basis"]
    return refs
