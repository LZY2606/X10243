"""服务层：把存储、积分、拟合串成可测试的业务操作。

所有操作产生机器可读事件；结果记录输入摘要、参数版本与父级依赖；
材料修订只让依赖它的结果过期，绝不回改历史。
"""

import json

from . import binding, integrate
from .outcomes import OK, diag
from .store import STREAM_KINDS


def _material_parents(store, streams):
    parents = []
    materials = {}
    for s in streams:
        m = store.get_material(s)
        if m is None:
            continue
        materials[s] = m
        parents.append({"type": "material", "stream": s, "version": m["version"],
                        "digest": m["digest"]})
    return materials, parents


def _update_event_payload(store, seq, **fields):
    row = store.db.execute("SELECT payload FROM events WHERE seq=?", (seq,)).fetchone()
    payload = json.loads(row["payload"])
    payload.update(fields)
    store.db.execute("UPDATE events SET payload=? WHERE seq=?",
                     (json.dumps(payload, ensure_ascii=False), seq))
    store.db.commit()


def import_material(store, stream, payload, branch="main", note=None):
    """导入或修订一条材料流（分开版本化，只增不改）。"""
    if stream not in STREAM_KINDS:
        ev = store.append_event("material.reject", {
            "stream": stream,
            "diagnostics": [diag("unknown_stream", stream,
                                 f"未知材料流: {stream}")]}, branch=branch)
        return {"status": "semantic_unsolvable", "event": ev}
    existed = store.get_material(stream) is not None
    ev = store.append_event(
        "material.revise" if existed else "material.import",
        {"stream": stream, "note": note}, branch=branch)
    version = store.put_material(stream, payload, ev["seq"])
    _update_event_payload(store, ev["seq"], version=version)
    ev["payload"]["version"] = version
    staled = _mark_stale_dependents(store, stream, version, ev["seq"], branch)
    return {"status": OK, "event": ev, "version": version, "staled": staled}


def _mark_stale_dependents(store, stream, new_version, seq, branch):
    """新材料版本使直接依赖旧版的结果过期（级联），原结果不改写。"""
    staled = []
    changed = True
    while changed:
        changed = False
        for r in store.results():
            if r["id"] in staled:
                continue
            status, _ = store._status_asof(r["id"], None)
            if status in ("stale", "rejected"):
                continue
            hit = None
            for p in r["parents"]:
                if p.get("type") == "material" and p.get("stream") == stream \
                        and p.get("version", 0) < new_version:
                    hit = f"{stream}_revised_to_v{new_version}"
                if p.get("type") == "result" and p.get("id") in staled:
                    hit = f"parent_result_stale:{p['id']}"
            if hit:
                store.set_result_status(r["id"], "stale", hit, seq)
                store.append_event("result.status", {
                    "result_id": r["id"], "status": "stale", "reason": hit,
                    "message": "材料修订使结果过期，原结果保留不改写"}, branch=branch)
                staled.append(r["id"])
                changed = True
    return staled


def run_integration(store, params, branch="main"):
    materials, parents = _material_parents(
        store, ["power", "injections", "baseline", "blank"])
    outcome = integrate.run_integration(materials, params)
    payload = {"params": params, "status": outcome["status"],
               "diagnostics": outcome["diagnostics"]}
    ev = store.append_event("integrate.run", payload,
                            parents=[p.get("digest", "") for p in parents],
                            branch=branch)
    if outcome["status"] == OK:
        rid = store.put_result("integration", outcome["result"], parents,
                               params, ev["seq"])
        _update_event_payload(store, ev["seq"], result_id=rid)
        outcome["result_id"] = rid
    outcome["event"] = ev
    return outcome


def run_fit(store, budget=None, integration_id=None, branch="main"):
    ints = store.results(kind="integration")
    if integration_id:
        ints = [r for r in ints if r["id"] == integration_id]
    ints = [r for r in ints if r["status"] in ("candidate", "accepted")]
    if not ints:
        ev = store.append_event("fit.run", {
            "status": "semantic_unsolvable",
            "diagnostics": [diag("no_integration", "fit",
                                 "没有可用积分结果")]}, branch=branch)
        return {"status": "semantic_unsolvable",
                "diagnostics": [{"code": "no_integration",
                                 "path": "fit", "message": "没有可用积分结果"}],
                "event": ev}
    integ = ints[-1]
    materials, mparents = _material_parents(store, ["concentration", "cell_volume"])
    if "concentration" not in materials or "cell_volume" not in materials:
        ev = store.append_event("fit.run", {
            "status": "semantic_unsolvable",
            "diagnostics": [diag("material_missing", "concentration",
                                 "缺少浓度或池体积材料")]}, branch=branch)
        return {"status": "semantic_unsolvable",
                "diagnostics": [{"code": "material_missing",
                                 "path": "concentration",
                                 "message": "缺少浓度或池体积材料"}], "event": ev}
    conc = materials["concentration"]["payload"]
    cond = {"cell_conc_M": float(conc["cell_M"]),
            "syringe_conc_M": float(conc["syringe_M"]),
            "cell_volume_uL": float(materials["cell_volume"]["payload"]["volume_uL"])}
    params = {"budget": budget or {"max_iter": 200},
              "models": list(binding.MODEL_PARAMS)}
    outcome = binding.fit_models(integ["payload"], cond, budget=budget)
    parents = mparents + [{"type": "result", "id": integ["id"],
                           "digest": integ["input_digest"]}]
    payload = {"params": params, "status": outcome["status"],
               "diagnostics": outcome["diagnostics"],
               "integration_id": integ["id"]}
    ev = store.append_event("fit.run", payload, branch=branch)
    candidate_ids = []
    if outcome["status"] == OK:
        shared = {"observed_uJ": outcome["result"]["observed_uJ"],
                  "conditions": outcome["result"]["conditions"],
                  "units": outcome["result"]["units"]}
        for cand in outcome["result"]["candidates"]:
            rid = store.put_result(
                "fit_candidate", {**cand, **shared}, parents, params, ev["seq"])
            candidate_ids.append(rid)
        _update_event_payload(store, ev["seq"], candidate_ids=candidate_ids)
        outcome["candidate_ids"] = candidate_ids
    outcome["event"] = ev
    return outcome


def decide(store, action, target_id, rationale, branch="main", scope="binding_model"):
    """接受 / 否定 / 参数调整。冲突决定全部保留并互相链接。"""
    target = store.get_result(target_id)
    if target is None:
        ev = store.append_event("decision.reject_noop", {
            "target_id": target_id,
            "diagnostics": [diag("target_missing", target_id,
                                 "决定目标不存在")]}, branch=branch)
        return {"status": "semantic_unsolvable", "event": ev}
    conflicts = []
    if action == "accept":
        for r in store.results(kind=target["kind"]):
            if r["id"] == target_id:
                continue
            if r["status"] == "accepted":
                conflicts.append(r["id"])
    ev = store.append_event(f"decision.{action}", {
        "target_id": target_id, "target_kind": target["kind"],
        "scope": scope, "rationale": rationale,
        "conflicts_with": conflicts,
        "message": "人工决定；冲突双方均保留，可回放"}, branch=branch)
    status = {"accept": "accepted", "reject": "rejected",
              "adjust": "adjusted"}[action]
    store.set_result_status(target_id, status, f"decision:{ev['id']}", ev["seq"])
    return {"status": OK, "event": ev, "conflicts_with": conflicts}
