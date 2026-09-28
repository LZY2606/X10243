"""事件溯源引擎：材料版本化 + 派生层 + 可重放判定事件。

事件（机器可读取 JSON）：
- MATERIAL_ADDED     新材料或材料新版本（不可变层）
- INTEGRALS_DERIVED  注射积分（记录输入摘要/规则版本/父依赖）
- BLANK_APPLIED      稀释热扣除（显式对应规则）
- MODELS_DERIVED     绑定模型候选（保留全部合法候选与等优标记）
- MODELS_INVALIDATED 浓度等修订使模型过期（但不改原积分）
- DECISION           人工接受/否定/参数调整（可冲突）
- NOTE/CHECKOUT      分支与历史回看

重放即重建：回到任意旧事件恢复当时依赖版本与可见结论。
"""
from __future__ import annotations

import json
import hashlib

from .errors import SemanticInvalid, Diagnostic
from .units import require, Quantity
from .model import Material
from . import integration as integ
from . import blank as blankmod
from . import binding as bind


def digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _event(seq, etype, payload, branch="main", parents=None, decision=None):
    eid = f"E{seq:04d}"
    return {
        "id": eid,
        "seq": seq,
        "type": etype,
        "branch": branch,
        "parents": parents or [],
        "payload": payload,
        "digest": "",
        "decision": decision,
    }


def _qval(qdict):
    return qdict.get("value")


def build_protocol(protocol_material: Material):
    if protocol_material is None:
        return []
    out = []
    for inj in protocol_material.data["injections"]:
        t = _qval(inj["time_q"])
        d = _qval(inj["duration_q"])
        v = _qval(inj["volume_q"])
        if None in (t, d, v):
            raise SemanticInvalid(
                f"注射 {inj['index']} 的时间/时长/体积存在无法转换的值",
                [Diagnostic("unconvertible", "保留原值，禁止默认值",
                            f"protocol inj{inj['index']}")])
        out.append({"index": inj["index"], "time": t, "duration": d,
                    "volume": v,
                    "raw_time": inj["time_q"]["raw"],
                    "raw_volume": inj["volume_q"]["raw"]})
    out.sort(key=lambda x: x["index"])
    return out


def build_blank_protocol(material: Material):
    if material is None:
        return None
    return build_protocol(material)


def _concentration_map(material: Material):
    if material is None:
        return None
    m = {}
    for row in material.data["values"]:
        v = _qval(row["concentration_q"])
        if v is None:
            raise SemanticInvalid(
                f"{material.kind} 第 {row['index']} 行浓度无法转换",
                [Diagnostic("unconvertible", "浓度原值保留",
                            f"{material.kind} idx{row['index']}")])
        m[row["index"]] = v
    return m


def _latest(state, kind):
    versions = state["materials"].get(kind)
    return versions[-1] if versions else None


# ----------------------------------------------------------------------
# 派生：仅基于给定材料快照计算，不写状态
# ----------------------------------------------------------------------

def derive_integrals(state, discard_first=True):
    power = _latest(state, "power")
    protocol = build_protocol(_latest(state, "protocol"))
    baseline = _latest(state, "baseline")
    if power is None:
        raise SemanticInvalid("缺少 power 材料，无法积分")
    if not protocol:
        raise SemanticInvalid("缺少 protocol 材料，无法确定注射区间")
    points = integ.integrate(power, protocol, baseline,
                             discard_first=discard_first)
    return points, power


def derive_blank(state, sample_points):
    blank_curve = _latest(state, "blank")
    blank_proto_mat = _latest(state, "blank_protocol")
    if blank_curve is None:
        return None, None
    blank_proto = build_protocol(blank_proto_mat) if blank_proto_mat \
        else build_protocol(_latest(state, "protocol"))
    baseline = _latest(state, "baseline")
    blank_points = integ.integrate(blank_curve, blank_proto, baseline,
                                   discard_first=True)
    power_id = _latest(state, "power").mid()
    corr = blankmod.apply_blank(
        sample_points, blank_points,
        power_id=power_id, blank_power_id=blank_curve.mid())
    return corr, blank_points


def derive_models(state, integrals, corrections):
    ligand = _concentration_map(_latest(state, "ligand"))
    analyte_mat = _latest(state, "analyte")
    cell = _latest(state, "cell")
    if ligand is None or cell is None:
        raise SemanticInvalid("缺少 ligand 浓度或 cell 池体积，无法拟合模型")
    V0 = _qval(cell.data["cell_volume_q"])
    if V0 is None:
        raise SemanticInvalid("池体积无法转换，禁止用默认值")
    m0 = None
    if analyte_mat is not None and analyte_mat.data["values"]:
        m0 = _qval(analyte_mat.data["values"][0]["concentration_q"])
    if m0 is None:
        m0 = 1e-3  # 被滴定物浓度仅作尺度；缺省时的处理改为语义错误更严格
        raise SemanticInvalid("缺少 analyte 初始浓度，无法拟合绑定模型")

    protocol = build_protocol(_latest(state, "protocol"))
    # 用空白校正后的热；缺失对应则回落到未校正热并在事件中标注
    heat_by_idx = {p.injection_index: p for p in integrals}
    corr_by_idx = {c.injection_index: c for c in corrections} \
        if corrections else {}
    x_k, heats, used_idx = [], [], []
    used_injections = []
    X_all = bind.concentrations(protocol, V0, ligand)
    M_all = bind.analyte_concentration(protocol, V0, m0)
    for k, inj in enumerate(protocol):
        idx = inj["index"]
        p = heat_by_idx.get(idx)
        if p is None or p.discarded or p.energy_j is None:
            continue
        c = corr_by_idx.get(idx)
        heat = p.energy_j
        if c is not None:
            if c.status == "corrected":
                heat = c.corrected_energy_j
            elif c.status == "no_match":
                continue  # 无对应空白的注不进入绑定拟合，避免静默偏置
        x_k.append(X_all[k])
        heats.append(heat)
        used_idx.append(idx)
        used_injections.append(inj)
    if len(heats) < 3:
        raise SemanticInvalid(
            f"可用于拟合的积分点仅 {len(heats)} 个，一位点至少需要 3 个")
    X = np_array(x_k)
    M = np_array([M_all[protocol.index(i)] for i in used_injections])
    fits = bind.fit_all(x_k, heats, X, M, used_injections, V0)
    return fits, used_idx, V0


def np_array(v):
    import numpy as np
    return np.array(v, dtype=float)


# ----------------------------------------------------------------------
# 状态 / 事件应用 / 重放
# ----------------------------------------------------------------------

def new_state():
    return {
        "materials": {},      # kind -> [Material(dict), ...] latest last
        "integrals": None,    # 最新积分结果快照（随依赖版本重算）
        "blanks": None,
        "models": None,
        "model_stale": False,
        "model_stale_reason": None,
        "decisions": [],
        "judgments": [],
        "head": {},           # branch -> event_id
        "tips": {},
        "checks": [],
    }


def _mat_from_event(payload):
    return payload["material"]


def _judgment(code, subject, status, detail, evidence):
    return {"code": code, "subject": subject, "status": status,
            "detail": detail, "evidence": evidence}


def _scan_judgments(points, corrections):
    js = []
    for p in points:
        if p.overlap_with:
            js.append(_judgment(
                "overlapping_peaks", f"inj{p.injection_index}", "allowed",
                ("注射后峰允许重叠：积分窗按相邻注射时间切分，"
                 f"与注射 {p.overlap_with} 共窗"),
                {"window": [p.win_start, p.win_end],
                 "overlap_with": p.overlap_with,
                 "rule": p.inputs["rule_version"]}))
        if p.discarded:
            js.append(_judgment(
                "first_injection_discard", f"inj{p.injection_index}",
                "discarded", p.reason,
                {"window": [p.win_start, p.win_end]}))
        if p.status == "crosses_gap":
            js.append(_judgment(
                "data_gap_blocks_window", f"inj{p.injection_index}",
                "blocked",
                "积分窗不跨越数据缺口：该注射积分作废，能量留空",
                {"gap_segments": p.gap_segments,
                 "window": [p.win_start, p.win_end]}))
        if p.status == "endpoint_failed":
            js.append(_judgment(
                "integration_endpoint", f"inj{p.injection_index}", "failed",
                p.reason, {"endpoints": p.endpoints}))
    if corrections:
        for c in corrections:
            if c.status == "no_match":
                js.append(_judgment(
                    "blank_correspondence_missing",
                    f"inj{c.injection_index}", "no_match",
                    c.reason, {"rule": c.rule,
                               "rule_version": c.rule_version}))
            if c.status == "corrected":
                js.append(_judgment(
                    "dilution_heat_rule", f"inj{c.injection_index}",
                    "applied",
                    f"按 {c.rule} 规则扣除空白（{c.rule_version}）",
                    {"blank_index": c.matched_blank_index,
                     "rule": c.rule,
                     "rule_version": c.rule_version}))
    return js


def _model_judgments(ranked):
    js = []
    eqs = [r for r in ranked if r.get("equivalent")]
    if len(eqs) >= 2:
        js.append(_judgment(
            "multiple_equivalent_models", "model-selection", "ambiguous",
            ("存在多个等优模型，全部保留为合法候选，不得静默单选："
             + ", ".join(sorted(e["model"] for e in eqs))),
            {"tie_tol": bind.AIC_TIE_TOL,
             "models": [{"model": e["model"], "aic": e["aic"]}
                        for e in eqs]}))
    for r in ranked:
        if r.get("budget_exhausted"):
            js.append(_judgment(
                "fit_budget_exhausted", r["model"], "budget",
                "拟合求值预算耗尽，与格式损坏/语义无解区分",
                {"evaluations": r["evaluations"]}))
        if not r.get("converged") and not r.get("budget_exhausted"):
            js.append(_judgment(
                "fit_not_converged", r["model"], "unconverged",
                "拟合未收敛", {}))
    return js


def add_material_event(state, material_dict, affects=None, seq=0,
                       branch="main", parents=None):
    kind = material_dict["kind"]
    versions = state["materials"].setdefault(kind, [])
    new_version = (versions[-1]["version"] + 1) if versions else 1
    material_dict["version"] = new_version
    material_dict["supersedes"] = versions[-1]["version"] if versions else None
    material_dict["affects"] = affects
    versions.append(material_dict)

    judgments = []
    # 浓度修订：使模型过期，但不改原积分
    if kind in ("ligand", "analyte", "cell") and state["integrals"] is not None:
        state["model_stale"] = True
        state["model_stale_reason"] = (
            f"{kind}@v{new_version} 修订：模型结果过期，需重算；"
            "原积分输入未变，保持不变")
        judgments.append(_judgment(
            "concentration_revision_invalidates_model",
            f"{kind}@v{new_version}", "stale",
            state["model_stale_reason"],
            {"supersedes": material_dict["supersedes"],
             "affected_integrals": "unchanged",
             "affected_models": "invalidated"}))
    # 功率局部改动：只重算受影响注射，其它积分身份与值保持
    scope_note = None
    if kind == "power" and state["integrals"] is not None:
        scope_note = ("power 新版本：仅受影响注射需要重新积分；"
                      f"局部依赖范围 {affects if affects else '全局'}")
        judgments.append(_judgment(
            "power_localized_change", f"power@v{new_version}",
            "localized", scope_note, {"affects": affects}))

    payload = {"material": material_dict, "affects": affects}
    ev = _event(seq, "MATERIAL_ADDED", payload, branch, parents)
    ev["payload"]["judgments"] = judgments
    state["judgments"].extend(judgments)
    return ev


def recompute_event(state, seq=0, branch="main", parents=None,
                    force=False):
    """重算积分/空白/模型（遵循当前材料版本）。"""
    points, power = derive_integrals(state)
    corrections, blank_points = derive_blank(state, points)
    judgments = _scan_judgments(points, corrections)
    models_payload = None
    try:
        ranked, used_idx, V0 = derive_models(state, points, corrections)
        judgments.extend(_model_judgments(ranked))
        models_payload = {"candidates": ranked, "used_injections": used_idx,
                          "cell_volume_l": V0, "ok": True}
    except SemanticInvalid as exc:
        models_payload = {"ok": False, "error": exc.to_dict(),
                          "candidates": []}
        judgments.append(_judgment(
            "model_semantic_invalid", "model-selection", "invalid",
            str(exc), {"error_kind": exc.kind,
                       "diagnostics": [d.to_dict() for d in exc.diagnostics]}))

    state["integrals"] = [p.to_dict() if hasattr(p, "to_dict") else p
                          for p in points]
    state["blanks"] = {
        "corrections": [c.to_dict() if hasattr(c, "to_dict") else c
                        for c in (corrections or [])],
        "blank_integrals": [p.to_dict() if hasattr(p, "to_dict") else p
                            for p in (blank_points or [])],
    }
    state["models"] = models_payload
    state["model_stale"] = False
    state["model_stale_reason"] = None
    state["judgments"].extend(judgments)
    snapshot = {
        "integrals": state["integrals"],
        "blanks": state["blanks"],
        "models": state["models"],
        "model_stale": False,
        "model_stale_reason": None,
    }
    payload = {
        "snapshot": snapshot,
        "integral_ids": [p["id"] for p in state["integrals"]],
        "correction_ids": [c["id"]
                           for c in state["blanks"]["corrections"]],
        "model_ok": models_payload["ok"],
        "input_summary": {
            "power": power.mid(),
            "baseline": (_latest(state, "baseline").mid()
                         if _latest(state, "baseline") else None),
            "blank": (_latest(state, "blank").mid()
                      if _latest(state, "blank") else None),
            "rule_versions": {
                "integral": integ.RULE_VERSION,
                "blank": blankmod.RULE_VERSION,
                "binding": bind.MODEL_RULE_VERSION,
                "aic_tie_tol": bind.AIC_TIE_TOL,
            },
        },
        "judgments": judgments,
    }
    return _event(seq, "DERIVED", payload, branch, parents)


def decision_event(state, action, subject, params=None, rationale="",
                   conflicts_with=None, seq=0, branch="main", parents=None):
    """接受 accept / 否定 reject / 参数调整 adjust。冲突保留双方。"""
    d = {"action": action, "subject": subject, "params": params or {},
         "rationale": rationale, "conflicts_with": conflicts_with,
         "branch": branch}
    state["decisions"].append(d)
    judgments = [_judgment(
        "human_decision", subject, action, rationale,
        {"params": params or {}, "conflicts_with": conflicts_with})]
    if conflicts_with:
        judgments.append(_judgment(
            "conflicting_decisions", subject, "conflict",
            f"与决定 {conflicts_with} 冲突；两个版本同时保留",
            {"kept_both": True}))
    state["judgments"].extend(judgments)
    ev = _event(seq, "DECISION", {"decision": d, "judgments": judgments},
                branch, parents, decision=d)
    return ev


# ----------------------------------------------------------------------
# 从事件列表完整重放
# ----------------------------------------------------------------------

def replay(events, create_derived=True):
    state = new_state()
    for ev in events:
        t = ev["type"]
        p = ev["payload"]
        if t == "MATERIAL_ADDED":
            mat = p["material"]
            kind = mat["kind"]
            versions = state["materials"].setdefault(kind, [])
            versions.append(mat)
            state["judgments"].extend(p.get("judgments", []))
            if kind in ("ligand", "analyte", "cell") \
                    and state["integrals"] is not None:
                state["model_stale"] = True
        elif t == "DERIVED":
            # 历史回看：恢复当时记录的依赖版本与结论，绝不用最新计算覆盖历史
            snap = p.get("snapshot")
            if snap:
                state["integrals"] = snap["integrals"]
                state["blanks"] = snap["blanks"]
                state["models"] = snap["models"]
                state["model_stale"] = snap.get("model_stale", False)
                state["model_stale_reason"] = snap.get("model_stale_reason")
            elif create_derived:
                _rematerialize_derived(state)
            state["judgments"].extend(p.get("judgments", []))
        elif t == "DECISION":
            state["decisions"].append(p["decision"])
            state["judgments"].extend(p.get("judgments", []))
        state["head"][ev["branch"]] = ev["id"]
    return state


def _rematerialize_derived(state):
    try:
        points, _ = derive_integrals(state)
        state["integrals"] = [p.to_dict() for p in points]
        corr, bp = derive_blank(state, points)
        state["blanks"] = {
            "corrections": [c.to_dict() for c in (corr or [])],
            "blank_integrals": [p.to_dict() for p in (bp or [])]}
        try:
            ranked, used, V0 = derive_models(state, points, corr)
            state["models"] = {"candidates": ranked,
                               "used_injections": used,
                               "cell_volume_l": V0, "ok": True}
        except SemanticInvalid as exc:
            state["models"] = {"ok": False, "error": exc.to_dict(),
                               "candidates": []}
    except SemanticInvalid:
        pass
