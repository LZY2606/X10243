"""把仅追加事件流投影成某个分支、某个时间点的审阅状态。

回到任意旧事件时恢复当时的依赖版本与可见结论，绝不用最新结果覆盖历史。
"""
from __future__ import annotations

from .canonicalid import canonical

STREAMS = ["power", "baseline", "schedule", "concentration", "conditions", "blank"]


def project(store, pid: str, branch: str = "main", upto_seq: int | None = None) -> dict:
    events = store.branch_events(pid, branch, upto_seq)
    current: dict[str, dict] = {s: None for s in STREAMS}
    integrations: list[dict] = []
    fits: list[dict] = []
    decisions: list[dict] = []
    model_decision = None
    for e in events:
        p = e["payload"]
        t = e["type"]
        if t == "material_committed":
            current[p["stream"]] = {
                "id": p["material_id"], "version": p["version"],
                "stream": p["stream"], "kind": p["kind"], "source": p["source"],
                "event_id": e["eid"], "seq": e["seq"]}
        elif t == "integration_run":
            integrations.append({**p, "event_id": e["eid"], "seq": e["seq"],
                                 "superseded": False, "stale": False,
                                 "stale_reasons": []})
        elif t == "fit_run":
            fits.append({**p, "event_id": e["eid"], "seq": e["seq"],
                         "superseded": False, "stale": False, "stale_reasons": []})
        elif t == "decision_recorded":
            decisions.append({**p, "event_id": e["eid"], "seq": e["seq"]})
        elif t == "model_decision":
            model_decision = {**p, "event_id": e["eid"], "seq": e["seq"]}
        elif t == "branch_created":
            pass

    # 后写入的同输入积分运行取代旧的（例如改窗后），但旧结果仍保留可见
    seen: dict[str, str] = {}
    for run in integrations:
        fp = canonical(run.get("input_fingerprint"))
        if fp in seen:
            older = seen[fp]
            for r in integrations:
                if r["event_id"] == older:
                    r["superseded"] = True
        seen[fp] = run["event_id"]

    latest_int = next((r for r in reversed(integrations) if not r["superseded"]),
                      None)
    # 模型过期判定：拟合依赖的材料版本 / 采纳积分与当前不一致
    for fit in fits:
        deps = fit.get("dependency_versions", {})
        for stream in ("concentration", "conditions", "blank", "power", "baseline",
                       "schedule"):
            want = deps.get(stream)
            now = current.get(stream)
            if want and now and want.get("id") != now["id"]:
                fit["stale"] = True
                fit["stale_reasons"].append(
                    f"{stream} 已从 v{want.get('version')} 修订为 v{now['version']}")
        if latest_int and fit.get("integration_run_id") and \
                fit["integration_run_id"] != latest_int["run_id"]:
            fit["stale"] = True
            fit["stale_reasons"].append("积分运行已被新的端点/决定运行取代")

    return {
        "pid": pid, "branch": branch, "upto_seq": upto_seq,
        "head_seq": events[-1]["seq"] if events else 0,
        "current": current,
        "integrations": integrations,
        "fits": fits,
        "decisions": decisions,
        "model_decision": model_decision,
        "events": events,
    }


def decision_map(snapshot: dict) -> dict:
    overrides, blank_map, discard = {}, {}, []
    blank_rule = "same_index"
    for d in snapshot["decisions"]:
        kind = d["decision"]
        if kind == "discard_injection":
            if d.get("discarded", True):
                discard.append(d["injection_index"])
            else:
                discard = [x for x in discard if x != d["injection_index"]]
        elif kind == "window_endpoint":
            overrides[str(d["injection_index"])] = {
                "start_si": d["start_si"], "end_si": d["end_si"]}
        elif kind == "blank_correspondence":
            blank_rule = d["rule"]
            if d.get("mapping"):
                m = d["mapping"]
                blank_map = {int(k): (v if isinstance(v, int) else v)
                             for k, v in (m.items() if isinstance(m, dict)
                                          else enumerate(m))}
        elif kind == "blank_rule_global":
            blank_rule = d["rule"]
    return {"window_overrides": overrides, "blank_map": blank_map,
            "discard": discard, "blank_rule": blank_rule}


def explain(store, pid: str, branch: str, event_id: str) -> dict:
    """从最终状态反查某事件由哪条材料/哪版规则/哪次决定造成。"""
    target = None
    for b in [branch] + [x for x in store.list_branches(pid) if x != branch]:
        e = store.get_event(event_id, b)
        if e:
            target, branch = e, b
            break
    if not target:
        return {"found": False}
    chain = []
    cur = target
    seen = set()
    while cur and cur["eid"] not in seen:
        seen.add(cur["eid"])
        chain.append({"event_id": cur["eid"], "seq": cur["seq"], "type": cur["type"],
                      "payload": cur["payload"]})
        cur = store.get_event(cur["parent"], branch) if cur["parent"] else None
    return {"found": True, "branch": branch, "event": target, "ancestors": chain,
            "deps": target.get("deps") and __import__("json").loads(target["deps"])}
