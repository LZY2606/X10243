"""应用服务层：业务规则唯一落点；FastAPI 与测试都调用这里。"""
from __future__ import annotations

import datetime as _dt

from .canonicalid import canonical, cid
from .integration import IntegrationParams, integrate_all
from .models import SemanticUnsolvable, assemble_titration, fit_all_models
from .replay import decision_map, project

STREAM_KIND = {"power": "power_trace", "baseline": "baseline_trace",
               "schedule": "injection_schedule", "concentration": "concentration_table",
               "conditions": "conditions_table", "blank": "blank_run"}


def now_iso() -> str:
    return _dt.datetime(2026, 9, 25, 12, 0, 0).isoformat() + "Z"


def commit_material(store, pid: str, branch: str, stream: str, payload: dict,
                    diagnostics: list[dict], source: str, note: str = "") -> dict:
    if stream not in STREAM_KIND:
        raise ValueError(f"unknown stream {stream}")
    put = store.put_material(pid, stream, STREAM_KIND[stream], source, payload,
                             diagnostics)
    if put["deduplicated"]:
        return {"material_id": put["id"], "version": put["version"], "event": None,
                "deduplicated": True}
    ev = store.append_event(pid, branch, "material_committed", {
        "stream": stream, "material_id": put["id"], "version": put["version"],
        "kind": STREAM_KIND[stream], "source": source,
        "diagnostics_count": len(diagnostics), "note": note},
        deps=[put["id"]])
    return {"material_id": put["id"], "version": put["version"], "event": ev,
            "deduplicated": False, "diagnostics": diagnostics}


def _versions(snapshot):
    return {k: v for k, v in snapshot["current"].items()}


def run_integration(store, pid: str, branch: str, lag_s: float | None = None,
                    width_s: float | None = None, gap_factor: float = 5.0,
                    blank_rule: str | None = None,
                    extra_discard: list[int] | None = None) -> dict:
    snap = project(store, pid, branch)
    cur = snap["current"]
    missing = [s for s in ("power", "schedule") if not cur[s]]
    if missing:
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": "missing_stream",
                                 "message": f"缺少必需材料流 {missing}",
                                 "location": f"branch:{branch}"}]}
    dm = decision_map(snap)
    if blank_rule is not None:
        dm["blank_rule"] = blank_rule
    discard = sorted(set(dm["discard"]) | set(extra_discard or []))
    prev_lag = snap["integrations"][-1]["input_summary"]["lag_s"] \
        if snap["integrations"] else 60.0
    prev_w = snap["integrations"][-1]["input_summary"]["width_s"] \
        if snap["integrations"] else 120.0
    params = IntegrationParams(
        lag_s=prev_lag if lag_s is None else lag_s,
        width_s=prev_w if width_s is None else width_s,
        gap_factor=gap_factor, blank_rule=dm["blank_rule"],
        blank_map={int(k): v for k, v in dm["blank_map"].items()},
        window_overrides={int(k): v for k, v in dm["window_overrides"].items()},
        discard_indices=discard)
    power = store.get_material(cur["power"]["id"])["payload"]
    baseline = store.get_material(cur["baseline"]["id"])["payload"] if cur["baseline"] else None
    schedule = store.get_material(cur["schedule"]["id"])["payload"]
    blank = None
    if cur["blank"]:
        bm = store.get_material(cur["blank"]["id"])
        pl = bm["payload"]
        blank = {"power": pl.get("power"), "schedule": pl.get("schedule")}
    result = integrate_all(power, baseline, schedule, params, blank,
                           versions=_versions(snap))
    fp = cid("infp", {
        "params": result["param_version"],
        "materials": {k: (v["id"] if v else None) for k, v in cur.items()}})
    ev = store.append_event(pid, branch, "integration_run", {
        "run_id": result["run_id"],
        "records": result["records"],
        "gaps_si": result["gaps_si"],
        "input_summary": result["input_summary"],
        "param_version": result["param_version"],
        "dependency_versions": {k: ({"id": v["id"], "version": v["version"]}
                                    if v else None) for k, v in cur.items()},
        "input_fingerprint": fp},
        deps=[cur["power"]["id"], cur["schedule"]["id"]]
        + ([cur["baseline"]["id"]] if cur["baseline"] else [])
        + ([cur["blank"]["id"]] if cur["blank"] else []))
    return {"ok": True, "event": ev, "run": result}


def record_decision(store, pid: str, branch: str, decision: str, rationale: str,
                    **fields) -> dict:
    payload = {"decision": decision, "rationale": rationale, **fields}
    return store.append_event(pid, branch, "decision_recorded", payload)


def run_fit(store, pid: str, branch: str, max_evals: int | None = None) -> dict:
    snap = project(store, pid, branch)
    cur = snap["current"]
    if not cur["concentration"] or not cur["conditions"]:
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": "missing_stream",
                                 "message": "拟合需要浓度表与实验条件（池体积）",
                                 "location": f"branch:{branch}"}]}
    latest_run = next((r for r in reversed(snap["integrations"])
                       if not r["superseded"]), None)
    if not latest_run:
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": "no_integration",
                                 "message": "还没有任何积分运行",
                                 "location": f"branch:{branch}"}]}
    concentration = store.get_material(cur["concentration"]["id"])["payload"]
    conditions = store.get_material(cur["conditions"]["id"])["payload"]
    try:
        data = assemble_titration(latest_run["records"], concentration, conditions)
    except SemanticUnsolvable as exc:
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": exc.reason.split(":")[0],
                                 "message": exc.reason, "location": exc.location}]}
    result = fit_all_models(data, max_evals=max_evals)
    if result["outcome"] != "converged":
        return {"ok": True, "outcome": "budget_exhausted", "result": result}
    ev = store.append_event(pid, branch, "fit_run", {
        "fit_id": cid("fitrun", {"run": latest_run["run_id"],
                                 "concentration": cur["concentration"]["id"],
                                 "candidates": [c["candidate_id"]
                                                for c in result["candidates"]]}),
        "integration_run_id": latest_run["run_id"],
        "candidates": result["candidates"],
        "tied_candidate_ids": result["tied_candidate_ids"],
        "unique_winner": result["unique_winner"],
        "used_indices": [int(v) for v in data["index"]],
        "budget_used": result["budget_used"],
        "dependency_versions": {k: ({"id": v["id"], "version": v["version"]}
                                    if v else None) for k, v in cur.items()}},
        deps=[latest_run["run_id"], cur["concentration"]["id"],
              cur["conditions"]["id"]])
    return {"ok": True, "outcome": "converged", "event": ev, "result": result}


def decide_model(store, pid: str, branch: str, action: str, candidate_id: str | None,
                 rationale: str) -> dict:
    if action not in ("accept", "reject_all"):
        raise ValueError("action must be accept|reject_all")
    snap = project(store, pid, branch)
    latest_fit = next((f for f in reversed(snap["fits"]) if not f["superseded"]), None)
    if not latest_fit:
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": "no_fit", "message": "还没有模型拟合",
                                 "location": f"branch:{branch}"}]}
    if action == "accept":
        ids = {c["candidate_id"]: c for c in latest_fit["candidates"]}
        if candidate_id not in ids:
            return {"ok": False, "outcome": "semantic_unsolvable",
                    "diagnostics": [{"code": "unknown_candidate",
                                     "message": "候选身份不在当前拟合中",
                                     "location": candidate_id or "null"}]}
    return {"ok": True, "event": store.append_event(
        pid, branch, "model_decision",
        {"action": action, "candidate_id": candidate_id, "rationale": rationale,
         "fit_id": latest_fit["fit_id"]},
        deps=[latest_fit["fit_id"]])}


def create_branch(store, pid: str, branch: str, rationale: str) -> dict:
    if branch in store.list_branches(pid):
        return {"ok": False, "outcome": "semantic_unsolvable",
                "diagnostics": [{"code": "branch_exists", "message": "分支已存在",
                                 "location": branch}]}
    head = store.branch_head(pid, "main")
    store.create_branch(pid, branch, head)
    ev = store.append_event(pid, branch, "branch_created",
                            {"name": branch, "rationale": rationale})
    return {"ok": True, "event": ev}
