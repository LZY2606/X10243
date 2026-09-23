"""复核视图层: 页面联动所需的读模型。

- 每个显示值都带稳定身份 (材料/事件/结果/候选 id)。
- 回到任意旧事件 (as_of) 时恢复当时的依赖版本与可见结论, 不用最新结果覆盖历史。
- 失效判定: 浓度修订使模型过期但不改原积分; 基线修订使依赖它的积分/扣除/拟合过期。
"""
from __future__ import annotations

from .units import normalize


def _version_map(store, as_of):
    """kind -> 截至 as_of 的最新材料版本。"""
    out = {}
    for m in store.materials(as_of=as_of):
        out[m["kind"]] = m["version"]
    return out


def staleness(store, result, as_of=None):
    """返回 (stale: bool, reasons: [str])。规则见模块 docstring。"""
    p = result["payload"]
    inputs = p.get("inputs", {})
    reasons = []
    current = _version_map(store, as_of)
    max_seq_by_kind = {}
    for r2 in store.results(as_of=as_of):
        max_seq_by_kind[r2["kind"]] = max(
            max_seq_by_kind.get(r2["kind"], 0), r2["seq"])

    refs = []

    def walk(value):
        if isinstance(value, dict):
            if "result_seq" in value:
                refs.append(("result", value))
            elif "version" in value and "kind" in value:
                refs.append(("material", value))
            else:
                for v in value.values():
                    walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(inputs)
    for ref_kind, ref in refs:
        if ref_kind == "material":
            cur = current.get(ref["kind"])
            if cur is not None and cur != ref["version"]:
                reasons.append(
                    f"{ref['kind']} 已修订 v{ref['version']}→v{cur}; "
                    f"该结果基于 v{ref['version']}")
        else:
            newer = max_seq_by_kind.get(ref["result_kind"], 0)
            if newer > ref["result_seq"]:
                reasons.append(
                    f"依赖的 {ref['result_kind']} 已有更新结果 "
                    f"(seq {ref['result_seq']}→{newer})")
    # 积分不依赖浓度: 浓度修订不会使积分过期 (规则: 不改原积分)
    if result["kind"] == "injection_integrals":
        reasons = [r for r in reasons if not r.startswith("concentrations")]
    return (len(reasons) > 0, reasons)


def review_state(store, as_of=None):
    head = store.head_seq()
    if as_of is None:
        as_of = head
    materials = store.materials(as_of=as_of)
    latest_by_kind = {}
    for m in materials:
        latest_by_kind[m["kind"]] = m
    events = store.events(as_of=as_of)
    results = store.results(as_of=as_of)

    mat_view = [{
        "id": m["id"], "kind": m["kind"], "version": m["version"],
        "digest": m["digest"], "seq": m["seq"],
        "current": latest_by_kind[m["kind"]]["id"] == m["id"],
        "supersedes": m["supersedes"],
    } for m in materials]

    res_view = []
    all_diags = []
    for r in results:
        stale, reasons = staleness(store, r, as_of=as_of)
        p = r["payload"]
        res_view.append({
            "id": r["id"], "kind": r["kind"], "seq": r["seq"],
            "event_id": r["event_id"], "status": r["status"],
            "digest": r["digest"], "stale": stale, "stale_reasons": reasons,
            "model": p.get("model"),
            "n_candidates": len(p.get("candidates", [])) or None,
            "summary": _result_summary(r),
        })
        for d in p.get("diagnostics", []):
            all_diags.append({**d, "source": r["id"]})

    evt_view = [{
        "id": e["id"], "seq": e["seq"], "kind": e["kind"],
        "status": e["status"], "parents": e["parents"],
        "param_version": e["param_version"],
        "summary": _event_summary(e),
    } for e in events]

    decisions = [e for e in evt_view if e["kind"] == "decision"]
    return {
        "as_of": as_of, "head_seq": head,
        "materials": mat_view,
        "events": evt_view,
        "results": res_view,
        "decisions": decisions,
        "diagnostics": all_diags,
    }


def _event_summary(e):
    p = e["payload"]
    k = e["kind"]
    if k == "import_material":
        return f"导入 {p.get('material_kind')} v{p.get('version')} ({p.get('material_id','')[:24]}…)"
    if k == "revise_material":
        return f"修订 {p.get('material_kind')} → v{p.get('version')}: {p.get('reason','')}"
    if k == "integrate":
        return f"注射积分 ({p.get('result_id','')[:20]}…) 状态 {p.get('status')}"
    if k == "blank_subtract":
        return f"空白扣除 规则={p.get('rule',{}).get('kind')} 状态 {p.get('status')}"
    if k == "fit":
        return f"拟合 {p.get('model')} 候选 {p.get('n_candidates')} 状态 {p.get('status')}"
    if k == "compare":
        return f"模型比较 等优={p.get('equally_good')}"
    if k == "decision":
        d = p.get("decision", {})
        return (f"决定[{d.get('reviewer','?')}] {d.get('action')} "
                f"{d.get('target','')[:26]} — {d.get('rationale','')}")
    if k == "scope_check":
        return f"范围核验[{p.get('rule_id')}] {'通过' if p.get('passed') else '失败'}"
    return k


def _result_summary(r):
    p = r["payload"]
    if r["kind"] == "injection_integrals":
        ok = sum(1 for i in p.get("injections", []) if i["status"] == "ok")
        return f"{ok}/{len(p.get('injections', []))} 针积分成功"
    if r["kind"] == "blank_subtraction":
        miss = sum(1 for e in p.get("series", {}).get("blank_corrected", [])
                   if e["status"] == "blank_missing")
        return f"规则 {p.get('rule',{}).get('kind')}, 缺空白 {miss} 针"
    if r["kind"] == "model_fit":
        if p.get("candidates"):
            c = p["candidates"][0]
            return f"{p.get('model')} 最优 RSS={c['rss']:.4g} 候选 {len(p['candidates'])}"
        return f"{p.get('model')} 无候选"
    if r["kind"] == "model_comparison":
        return f"等优模型 {p.get('equally_good')}"
    return r["status"]


def fit_view(store, fit_result_id, as_of=None):
    """拟合详情: 候选、残差、有效区间与失效原因, 全部带稳定身份。"""
    r = store.result(fit_result_id)
    if r is None:
        return None
    stale, reasons = staleness(store, r, as_of=as_of)
    p = r["payload"]
    return {
        "id": r["id"], "model": p.get("model"), "status": r["status"],
        "stale": stale, "stale_reasons": reasons,
        "units": p.get("units", {}),
        "diagnostics": p.get("diagnostics", []),
        "candidates": p.get("candidates", []),
        "inputs": p.get("inputs", {}),
        "event_id": r["event_id"],
    }
