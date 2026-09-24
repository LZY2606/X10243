"""服务管线：把当前材料版本串成 积分 → 稀释扣除 → 模型拟合。

每个计算记录输入摘要、参数版本与父级依赖；输入不完整时
返回带位置的诊断（semantic_unresolved / format_error），
不得用空值或默认值伪装成功。
"""
from __future__ import annotations

from . import binding, integrate, units
from .store import Store, digest_of

PARAM_VERSION = "pipeline/1.0"


def _current(store: Store, kind: str):
    return store.state_at()["materials"].get(kind)


def _norm_scalar_material(mat, kind, path):
    if mat is None:
        return None, {"path": path, "code": "material_missing",
                      "message": f"缺少材料 {path}"}
    value, diag = units.normalize_scalar(mat["raw"], kind, path)
    return value, (diag.to_json() if diag else None)


def compute_integrations(store: Store, trace_kind: str = "power_trace",
                         opts: dict | None = None, branch: str = "main"):
    """对指定功率曲线做注射积分。返回 (computation_id, created, result)。"""
    state = store.state_at()
    mats = state["materials"]
    trace_mat = mats.get(trace_kind)
    sched_mat = mats.get("injection_schedule")
    base_mat = mats.get("baseline")
    parents = {m["id"]: m["version"] for m in
               (trace_mat, sched_mat, base_mat) if m}
    missing = [k for k, m in (("trace", trace_mat), ("schedule", sched_mat))
               if m is None]
    if missing:
        result = {"status": "semantic_unresolved", "injections": [],
                  "diagnostics": [{"path": k, "code": "material_missing",
                                   "message": f"缺少{k}材料"} for k in missing],
                  "input_summary": {"missing": missing},
                  "rule_basis": integrate.RULE_BASIS}
        cid, created = store.put_computation(
            f"integration:{trace_kind}", PARAM_VERSION, parents,
            "semantic_unresolved", result, branch)
        return cid, created, result
    trace = units.normalize_trace(trace_mat["raw"], f"materials.{trace_kind}")
    sched = units.normalize_schedule(sched_mat["raw"], "materials.injection_schedule")
    base = (units.normalize_trace(base_mat["raw"], "materials.baseline")
            if base_mat else None)
    if "format_error" in sched:
        result = {"status": "format_error", "injections": [],
                  "diagnostics": sched["diagnostics"],
                  "input_summary": {"schedule": "format_error"},
                  "rule_basis": integrate.RULE_BASIS}
        cid, created = store.put_computation(
            f"integration:{trace_kind}", PARAM_VERSION, parents,
            "format_error", result, branch)
        return cid, created, result
    res = integrate.integrate(trace, sched["injections"], base, opts)
    res["input_summary"] = {
        "trace_points": len(trace.get("t", [])),
        "injections": len(sched["injections"]),
        "opts": opts or integrate.DEFAULT_OPTS,
        "basis_t": trace.get("basis_t"),
    }
    cid, created = store.put_computation(
        f"integration:{trace_kind}", PARAM_VERSION, parents,
        res["status"], res, branch)
    return cid, created, res


def compute_blank_subtraction(store: Store, rule: dict,
                              branch: str = "main"):
    """稀释热扣除：对应规则显式给出并进入事件。"""
    sample_id, _, sample = compute_integrations(store, "power_trace")
    blank_id, _, blank = compute_integrations(store, "blank_trace")
    state = store.state_at()
    parents = {}
    for comp in state["computations"]:
        if comp["id"] in (sample_id, blank_id):
            parents[comp["id"]] = 0  # 计算间依赖用 id 记录，版本恒 0
    param_version = PARAM_VERSION + "|rule:" + digest_of(rule)[:8]
    res = integrate.subtract_blank(sample, blank, rule)
    res["input_summary"] = {"sample": sample_id, "blank": blank_id,
                            "rule": rule}
    cid, created = store.put_computation(
        "dilution_subtraction", param_version, parents, res["status"],
        res, branch)
    return cid, created, res


def compute_model_fit(store: Store, rule: dict | None = None,
                      budget: int = 20000, branch: str = "main"):
    """绑定模型拟合：依赖稀释扣除结果 + 浓度 + 池体积 + 注射时序。"""
    rule = rule or {"matching": "index"}
    sub_id, _, sub = compute_blank_subtraction(store, rule, branch)
    state = store.state_at()
    mats = state["materials"]
    conc, d1 = _norm_scalar_material_pair(mats.get("concentration"))
    vol, d2 = _norm_scalar_material(mats.get("cell_volume"), "volume",
                                    "materials.cell_volume")
    sched = units.normalize_schedule(mats["injection_schedule"]["raw"],
                                     "materials.injection_schedule")
    diags = [d for d in (d1, d2) if d] + sched.get("diagnostics", [])
    parents = {sub_id: 0}
    for k in ("concentration", "cell_volume", "injection_schedule"):
        if k in mats:
            parents[mats[k]["id"]] = mats[k]["version"]
    if conc is None or vol is None or "format_error" in sched:
        result = {"status": "semantic_unresolved", "candidates": [],
                  "diagnostics": diags or [{"path": "materials",
                                            "code": "input_incomplete",
                                            "message": "浓度或池体积缺失"}],
                  "input_summary": {"subtraction": sub_id},
                  "rule_basis": {"rule": "binding_model_candidates"}}
        cid, created = store.put_computation(
            "model_fit", PARAM_VERSION, parents, "semantic_unresolved",
            result, branch)
        return cid, created, result
    if sub.get("status") not in ("ok", "semantic_unresolved") or not sub.get("heats"):
        result = {"status": sub.get("status", "semantic_unresolved"),
                  "candidates": [],
                  "diagnostics": sub.get("diagnostics", []),
                  "input_summary": {"subtraction": sub_id},
                  "rule_basis": {"rule": "binding_model_candidates"}}
        cid, created = store.put_computation(
            "model_fit", PARAM_VERSION, parents, result["status"], result,
            branch)
        return cid, created, result
    ctx = {"v0_L": vol / 1e6, "mt0_uM": conc["cell"], "xt_uM": conc["syringe"],
           "inj_volumes_uL": [i["volume_uL"] if i["volume_uL"] is not None else 0.0
                              for i in sched["injections"]]}
    res = binding.fit_models(ctx, sub["heats"], budget=budget)
    res["input_summary"] = {"subtraction": sub_id, "ctx": {
        "v0_L": ctx["v0_L"], "mt0_uM": ctx["mt0_uM"], "xt_uM": ctx["xt_uM"]}}
    cid, created = store.put_computation(
        "model_fit", PARAM_VERSION, parents, res["status"], res, branch)
    return cid, created, res


def _norm_scalar_material_pair(mat):
    """concentration 材料：{syringe, cell} 两个浓度。"""
    if mat is None:
        return None, {"path": "materials.concentration",
                      "code": "material_missing", "message": "缺少浓度材料"}
    raw = mat["raw"]
    syr, ds = units.normalize_scalar(raw.get("syringe"), "concentration",
                                     "materials.concentration.syringe")
    cel, dc = units.normalize_scalar(raw.get("cell"), "concentration",
                                     "materials.concentration.cell")
    diag = next((d.to_json() for d in (ds, dc) if d), None)
    if syr is None or cel is None:
        return None, diag
    return {"syringe": syr, "cell": cel}, diag
