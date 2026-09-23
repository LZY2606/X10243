"""离线案例包: 滴定热谱 — ITC 注射积分、稀释热与绑定模型审议。

案例叙事 (领域材料, 非通用表格):
  初始导入: 功率采样/注射体积和时间/浓度/温度/池体积/基线/空白曲线 (分开版本化)
  -> 注射积分 -> 空白扣除(说明对应规则) -> 首注舍弃 -> 一位点/多位点/稀释拟合
  -> 模型比较与等优候选 -> 浓度修订(模型过期, 原积分不变) -> 基线修订(局部重算)
  -> 互相冲突的人工决定(均保留)
  第二分支(反例档案): 从基线快照出发, 每次只改动一个直接来源, 核验影响范围。

重放 (replay) 实际调用积分、空白扣除与拟合引擎, 判定写为机器可读事件。
"""
from __future__ import annotations

import json

import numpy as np

from . import integrate as integ_mod
from . import blank as blank_mod
from . import fit as fit_mod
from .units import normalize as _normalize_unit
from .store import digest_of

CASE_FORMAT = "itc-casepack/1"

TRUE_KA_MM = 10.0           # mM^-1  (= 1e7 M^-1)
TRUE_DH = -40.0             # kJ/mol
TRUE_N = 1.0
CCELL, CSYR_TRUE = 0.5, 4.6  # mM (真实浓度 = 修订后的值; v1 的 5.0 是称量误差)
V0_UL = 200.0
TAU = 15.0                  # 峰衰减常数 s
N_INJ = 20
RUN_END = 3240.0


def _injection_times():
    times = [300.0 + 150.0 * i for i in range(N_INJ)]
    times[12] = times[11] + 40.0   # inj-11/12 峰重叠 (40 s 间隔)
    return times


def _schedule_payload(times):
    injections = []
    for i, t in enumerate(times):
        if i == 0:
            injections.append({"id": "inj-00", "t": {"value": t, "unit": "s"},
                               "volume": {"value": 0.4, "unit": "µL"},
                               "role": "preliminary"})
        elif i == 14:
            # 无法转换的体积单位: 留在原位置, 摩尔热派生层附诊断
            injections.append({"id": f"inj-{i:02d}", "t": {"value": t, "unit": "s"},
                               "volume": {"value": 0.16, "unit": "drops"}})
        else:
            injections.append({"id": f"inj-{i:02d}", "t": {"value": t, "unit": "s"},
                               "volume": {"value": 2.0, "unit": "µL"}})
    return {"run_id": "ITC-2026-LYS-07",
            "basis": {"time": "elapsed_s_from_run_start"},
            "injections": injections}


def _true_heats(times):
    vols = [0.4] + [2.0] * (N_INJ - 1)
    gen_points = [{"vol_uL": v} for v in vols]
    q = fit_mod.predict_uJ("one_site", [TRUE_N, np.log10(TRUE_KA_MM), TRUE_DH],
                           gen_points, CCELL, CSYR_TRUE, V0_UL)
    rng = np.random.RandomState(20260919)
    blank_heat = np.array([6.0 - 0.08 * i for i in range(N_INJ)])
    observed = np.array(q) + blank_heat + rng.normal(0, 0.05, N_INJ)
    return [float(v) for v in q], [float(v) for v in observed]


def _power_payload(times, observed, extra_gap=None):
    ts = np.arange(0.0, RUN_END + 0.01, 2.0)
    rng = np.random.RandomState(20260920)
    gaps = [{"start": {"value": 1400.0, "unit": "s"},
             "end": {"value": 1420.0, "unit": "s"},
             "reason": "instrument_transfer"}]
    if extra_gap:
        gaps.append(extra_gap)
    powers = []
    for t in ts:
        if any(g["start"]["value"] <= t < g["end"]["value"] for g in gaps):
            continue
        p = 2.0 + 0.0002 * t
        for i, ti in enumerate(times):
            if t >= ti:
                p += observed[i] / TAU * np.exp(-(t - ti) / TAU)
        p += rng.normal(0, 0.08)
        powers.append({"t": {"value": round(float(t), 2), "unit": "s"},
                       "power": {"value": round(float(p), 5), "unit": "µJ/s"}})
    return {"run_id": "ITC-2026-LYS-07",
            "basis": {"time": "elapsed_s_from_run_start"},
            "sampling_hz": 0.5, "samples": powers, "gaps": gaps}


def _blank_payload():
    rng = np.random.RandomState(20260921)
    entries = []
    for i in range(18):  # 只有 18 条: inj-18/19 空白对应缺失
        vol = 0.4 if i == 0 else 2.0
        heat = 6.0 - 0.08 * i + rng.normal(0, 0.05)
        if i == 3:
            entries.append({"index": i, "volume": {"value": vol, "unit": "µL"},
                            "heat": {"value": round(heat / 4.184, 5),
                                     "unit": "µcal"}})
        elif i == 7:
            # 无法转换的热单位: 留在原位置附诊断, 对应按缺失处理
            entries.append({"index": i, "volume": {"value": vol, "unit": "µL"},
                            "heat": {"value": 1.4, "unit": "arbitrary"}})
        else:
            entries.append({"index": i, "volume": {"value": vol, "unit": "µL"},
                            "heat": {"value": round(heat, 5), "unit": "µJ"}})
    return {"run_id": "BLANK-2026-LYS-07", "entries": entries}


def build_casepack():
    """构造完整离线案例包 (材料初始版本 + 事件脚本)。"""
    times = _injection_times()
    _, observed = _true_heats(times)
    power1 = _power_payload(times, observed)
    power2 = _power_payload(times, observed, extra_gap={
        "start": {"value": 1048.0, "unit": "s"},
        "end": {"value": 1060.0, "unit": "s"},
        "reason": "counterexample_gap_over_injection"})
    sched = _schedule_payload(times)
    conc1 = {"syringe": {"value": 5.0, "unit": "mM", "label": "配体"},
             "cell": {"value": 0.5, "unit": "mM", "label": "蛋白"}}
    conc2 = {"syringe": {"value": 4.6, "unit": "mM", "label": "配体(重新称量)"},
             "cell": {"value": 0.5, "unit": "mM", "label": "蛋白"}}
    base1 = {"kind": "constant",
             "power": {"value": 2.0, "unit": "µJ/s"}}
    base2 = {"kind": "curve", "points": [
        {"t": {"value": 0.0, "unit": "s"}, "power": {"value": 2.0, "unit": "µJ/s"}},
        {"t": {"value": 1600.0, "unit": "s"}, "power": {"value": 2.32, "unit": "µJ/s"}},
        {"t": {"value": 3240.0, "unit": "s"}, "power": {"value": 2.648, "unit": "µJ/s"}},
    ]}
    temp = {"value": 25.0, "unit": "°C", "basis": "chiller_setpoint"}
    cellvol = {"value": 0.2, "unit": "mL"}
    blankm = _blank_payload()

    ops = [
        # ---------- 初始导入: 七类材料分开版本化 ----------
        {"op": "import", "material": {"kind": "power_curve", "payload": power1}, "as": "power1"},
        {"op": "import", "material": {"kind": "injection_schedule", "payload": sched}, "as": "sched1"},
        {"op": "import", "material": {"kind": "concentrations", "payload": conc1}, "as": "conc1"},
        {"op": "import", "material": {"kind": "temperature", "payload": temp}, "as": "temp1"},
        {"op": "import", "material": {"kind": "cell_volume", "payload": cellvol}, "as": "cellvol1"},
        {"op": "import", "material": {"kind": "baseline", "payload": base1}, "as": "base1"},
        {"op": "import", "material": {"kind": "blank_curve", "payload": blankm}, "as": "blankm1"},
        {"op": "milestone", "name": "imported",
         "label": "初始导入: 核对功率采样/注射时序/浓度/温度/池体积/基线/空白曲线"},
        # ---------- 积分、空白扣除、首注舍弃、拟合 ----------
        {"op": "integrate", "power": "power1", "schedule": "sched1", "baseline": "base1", "as": "integ1"},
        {"op": "blank_subtract", "integrals": "integ1", "blank": "blankm1",
         "rule": {"kind": "by_index"}, "as": "blank1"},
        {"op": "decide", "as_name": "discard0",
         "decision": {"reviewer": "A. Sato", "action": "discard_injection",
                      "target": "inj-00", "rationale": "首注体积0.4µL, 首注仪器伪影, 舍入拟合"}},
        {"op": "fit", "model": "one_site", "integrals": "integ1", "blank": "blank1",
         "concentrations": "conc1", "cell_volume": "cellvol1", "as": "fit1_1s"},
        {"op": "fit", "model": "two_site", "integrals": "integ1", "blank": "blank1",
         "concentrations": "conc1", "cell_volume": "cellvol1", "as": "fit1_2s"},
        {"op": "fit", "model": "dilution_constant", "integrals": "integ1", "blank": "blank1",
         "concentrations": "conc1", "cell_volume": "cellvol1", "as": "fit1_dil"},
        {"op": "compare", "fits": ["fit1_1s", "fit1_2s", "fit1_dil"], "as": "cmp1"},
        # ---------- 互相冲突的人工决定(第一回合) ----------
        {"op": "decide",
         "decision": {"reviewer": "A. Sato", "action": "accept",
                      "target_result": "fit1_1s", "candidate": 0,
                      "rationale": "一位点 RSS 最低, 残差无结构, 接受"}},
        # ---------- 一次参数修订: 浓度 -> 模型过期, 原积分不变 ----------
        {"op": "revise", "kind": "concentrations", "payload": conc2,
         "supersedes": "conc1", "reason": "重新称量, 注射器浓度 5.0->4.6 mM", "as": "conc2"},
        {"op": "fit", "model": "one_site", "integrals": "integ1", "blank": "blank1",
         "concentrations": "conc2", "cell_volume": "cellvol1", "as": "fit2_1s"},
        # ---------- 一次只影响局部依赖的源数据改动: 基线修订 ----------
        {"op": "revise", "kind": "baseline", "payload": base2,
         "supersedes": "base1", "reason": "基线含线性漂移, 改用分段曲线基线", "as": "base2"},
        {"op": "integrate", "power": "power1", "schedule": "sched1", "baseline": "base2", "as": "integ2"},
        {"op": "blank_subtract", "integrals": "integ2", "blank": "blankm1",
         "rule": {"kind": "by_index"}, "as": "blank2"},
        {"op": "fit", "model": "one_site", "integrals": "integ2", "blank": "blank2",
         "concentrations": "conc2", "cell_volume": "cellvol1", "as": "fit3_1s"},
        {"op": "fit", "model": "two_site", "integrals": "integ2", "blank": "blank2",
         "concentrations": "conc2", "cell_volume": "cellvol1", "as": "fit3_2s"},
        {"op": "fit", "model": "dilution_constant", "integrals": "integ2", "blank": "blank2",
         "concentrations": "conc2", "cell_volume": "cellvol1", "as": "fit3_dil"},
        {"op": "compare", "fits": ["fit3_1s", "fit3_2s", "fit3_dil"], "as": "cmp2"},
        # ---------- 互相冲突的人工决定(第二回合, 均保留) ----------
        {"op": "decide",
         "decision": {"reviewer": "M. Chen", "action": "reject",
                      "target_result": "fit1_1s", "candidate": 0,
                      "rationale": "浓度与基线均已修订, 原一位点结果过期"}},
        {"op": "decide",
         "decision": {"reviewer": "M. Chen", "action": "accept",
                      "target_result": "fit3_2s", "candidate": 0,
                      "rationale": "漂移基线修正后多位点候选同样优, 采纳多位点"}},
        {"op": "milestone", "name": "main_complete",
         "label": "主叙事完成: 一位点/多位点/稀释假设比较与冲突决定"},
        # ---------- 第二分支: 反例档案 (每次只改一个直接来源) ----------
        {"op": "milestone", "name": "counterexample_start",
         "label": "反例档案: 基线快照后逐项单源改动"},
        {"op": "scope_check", "rule_id": "archive_baseline",
         "statement": "功率采样/注射时序/浓度/温度/池体积/基线/空白曲线 七类材料分开版本化的基线快照",
         "check": {"type": "snapshot"}},
        {"op": "scope_check", "rule_id": "overlap_peaks_allowed",
         "statement": "注射后峰可重叠: inj-11/inj-12 间隔40s, 两针均正常积分, 窗口相接",
         "check": {"type": "integral_status", "result": "integ1",
                   "injections": ["inj-11", "inj-12"], "expect": "ok"}},
        {"op": "scope_check", "rule_id": "first_injection_discard",
         "statement": "首注舍弃: 原积分保留在结果中, 但不进入拟合输入",
         "check": {"type": "first_injection", "integrals": "integ1",
                   "fit": "fit1_1s", "injection": "inj-00"}},
        {"op": "scope_check", "rule_id": "blank_correspondence_missing",
         "statement": "空白对应缺失: by_index 规则下 inj-07(单位不可读)/inj-18/inj-19 缺对应, 诊断带位置, 原始序列保留",
         "check": {"type": "blank_missing", "result": "blank1",
                   "expect": [7, 18, 19]}},
        {"op": "scope_check", "rule_id": "concentration_revision_scope",
         "statement": "浓度修订使模型过期但不改原积分: 积分结果未失效且未重算, 旧拟合标记过期",
         "check": {"type": "concentration_scope",
                   "integrals": "integ1", "stale_fits": ["fit1_1s", "fit1_2s", "fit1_dil"]}},
        {"op": "scope_check", "rule_id": "multiple_equal_models",
         "statement": "多个等优模型: 容差内候选全部保留且稳定排序; 比较层给出等优集合",
         "check": {"type": "equal_models", "compare": "cmp2"}},
        {"op": "scope_check", "rule_id": "gap_and_endpoint",
         "statement": "数据缺口与积分端点: inj-07 窗在缺口截断, 末针窗在数据末端截断",
         "check": {"type": "flags", "result": "integ1",
                   "expect": {"inj-07": ["truncated_by_gap"],
                              "inj-19": ["truncated_at_end"]}}},
        {"op": "revise", "kind": "power_curve", "payload": power2,
         "supersedes": "power1", "reason": "反例: 注射点落入数据缺口", "as": "power2"},
        {"op": "integrate", "power": "power2", "schedule": "sched1", "baseline": "base2",
         "as": "integ_ce"},
        {"op": "scope_check", "rule_id": "injection_in_gap",
         "statement": "注射落入缺口: inj-05 不可积并给出带位置诊断, 不是空值; 其余针不受影响",
         "check": {"type": "injection_in_gap", "result": "integ_ce",
                   "injection": "inj-05"}},
        {"op": "scope_check", "rule_id": "single_source_change_scope",
         "statement": "只改功率曲线一个直接来源: 旧积分/旧拟合身份不变, 仅新增派生结果",
         "check": {"type": "unchanged_results",
                   "results": ["integ1", "integ2", "fit1_1s", "fit3_2s"]}},
        {"op": "blank_subtract", "integrals": "integ2", "blank": "blankm1",
         "rule": {"kind": "by_volume", "tolerance_uL": 0.5}, "as": "blank_ce"},
        {"op": "scope_check", "rule_id": "blank_rule_must_be_stated",
         "statement": "稀释热扣除必须说明对应规则: by_index 与 by_volume 结果并存, 规则写入事件",
         "check": {"type": "blank_rules", "results": ["blank1", "blank_ce"],
                   "expect_rules": ["by_index", "by_volume"]}},
        {"op": "integrate", "power": "power1", "schedule": "sched1", "baseline": "base1",
         "params": {"max_samples": 10}, "as": "integ_budget"},
        {"op": "scope_check", "rule_id": "budget_exhausted_distinct",
         "statement": "资源预算耗尽是独立状态, 不同于格式损坏与语义无解",
         "check": {"type": "result_status", "result": "integ_budget",
                   "expect": "budget_exhausted"}},
        {"op": "milestone", "name": "counterexample_complete",
         "label": "反例档案完成: 影响范围核验全部通过"},
    ]

    return {
        "format": CASE_FORMAT,
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": [
            "初始导入: 核对功率采样、注射体积和时间、浓度、温度、池体积、基线和空白曲线(分开版本化)",
            "注射积分: 重叠注射峰、数据缺口与积分端点按规则处理",
            "空白扣除: 说明对应规则(by_index), 缺对应处保留诊断与原始序列",
            "首注舍弃: inj-00 积分保留但不进入拟合",
            "模型审议: 比较一位点、多位点与稀释假设, 等优候选全部保留",
            "冲突决定: 一位点被接受又被拒绝, 两次决定都保留",
            "浓度修订: 5.0→4.6 mM, 模型过期, 原积分不变",
            "基线修订: 只影响依赖基线的积分与下游, 其他结果不变",
            "反例档案: 每次只改一个直接来源, 机器可读事件核验影响范围",
        ],
        "default_milestone": "main_complete",
        "ops": ops,
    }


# ---------------- 重放引擎 ----------------
from .store import MATERIAL_KINDS  # noqa: E402
from . import views  # noqa: E402


def _result_id(kind, payload):
    return f"res/{kind}/{digest_of({'kind': kind, 'payload': payload})[7:19]}"


def replay(store, pack):
    """把案例包事件脚本重放进 store, 实际调用积分/空白/拟合引擎。"""
    assert pack.get("format") == CASE_FORMAT
    syms = {}
    discarded = set()
    decision_events = []
    for op in pack["ops"]:
        kind = op["op"]
        if kind in ("import", "revise"):
            _op_import(store, syms, op, revise=(kind == "revise"))
        elif kind == "integrate":
            _op_integrate(store, syms, op)
        elif kind == "blank_subtract":
            _op_blank(store, syms, op)
        elif kind == "fit":
            _op_fit(store, syms, op, discarded, decision_events)
        elif kind == "compare":
            _op_compare(store, syms, op)
        elif kind == "decide":
            _op_decide(store, syms, op, discarded, decision_events)
        elif kind == "milestone":
            with store.operation("milestone", {"name": op["name"]}):
                store.put_event("milestone",
                                {"name": op["name"], "label": op.get("label", "")},
                                param_version="casepack/1.0")
        elif kind == "scope_check":
            _op_scope_check(store, syms, op)
        else:
            raise ValueError(f"未知操作 {kind}")
    return syms


def _op_import(store, syms, op, revise):
    if revise:
        m_kind, payload = op["kind"], op["payload"]
        prev = syms[op["supersedes"]]
        supersedes = prev["id"]
        reason = op.get("reason", "")
        evt_kind = "revise_material"
    else:
        spec = op["material"]
        m_kind, payload = spec["kind"], spec["payload"]
        supersedes, reason, evt_kind = None, "", "import_material"
    with store.operation(evt_kind, {"kind": m_kind}):
        m = store.put_material(m_kind, payload, supersedes=supersedes)
        store.put_event(evt_kind, {
            "material_id": m["id"], "material_kind": m["kind"],
            "version": m["version"], "reason": reason,
            "supersedes": supersedes},
            parents=[supersedes] if supersedes else [],
            input_digest=m["digest"], param_version="import/1.0")
    if "as" in op:
        syms[op["as"]] = m


def _op_integrate(store, syms, op):
    power, sched, base = (syms[op["power"]], syms[op["schedule"]],
                          syms[op["baseline"]])
    payload = integ_mod.integrate(power["payload"], sched["payload"],
                                  base["payload"], op.get("params"))
    payload["inputs"] = {
        "power_curve": {"id": power["id"], "kind": "power_curve",
                        "version": power["version"]},
        "injection_schedule": {"id": sched["id"], "kind": "injection_schedule",
                               "version": sched["version"]},
        "baseline": {"id": base["id"], "kind": "baseline",
                     "version": base["version"]},
    }
    res_id = _result_id("injection_integrals", payload)
    with store.operation("integrate"):
        evt = store.put_event("integrate", {
            "params": payload["params"], "status": payload["status"],
            "result_id": res_id},
            parents=[power["id"], sched["id"], base["id"]],
            input_digest=digest_of(payload["inputs"]),
            param_version=integ_mod.PARAM_VERSION, status=payload["status"])
        res = store.put_result("injection_integrals", evt["id"], payload,
                               status=payload["status"])
    syms[op["as"]] = res


def _op_blank(store, syms, op):
    integ, blankm = syms[op["integrals"]], syms[op["blank"]]
    payload = blank_mod.subtract(integ["payload"], blankm["payload"], op["rule"])
    payload["inputs"] = {
        "integrals": {"result_id": integ["id"],
                      "result_kind": "injection_integrals",
                      "result_seq": integ["seq"]},
        "blank_curve": {"id": blankm["id"], "kind": "blank_curve",
                        "version": blankm["version"]},
    }
    res_id = _result_id("blank_subtraction", payload)
    with store.operation("blank_subtract"):
        evt = store.put_event("blank_subtract", {
            "rule": payload["rule"], "status": payload["status"],
            "result_id": res_id},
            parents=[integ["id"], blankm["id"]],
            input_digest=digest_of(payload["inputs"]),
            param_version=blank_mod.RULE_VERSION, status=payload["status"])
        res = store.put_result("blank_subtraction", evt["id"], payload,
                               status=payload["status"])
    syms[op["as"]] = res


def _op_fit(store, syms, op, discarded, decision_events):
    integ = syms[op["integrals"]]
    blank_res = syms[op["blank"]] if op.get("blank") else None
    conc = syms[op["concentrations"]]
    cellvol = syms[op["cell_volume"]]
    v0, d = _normalize_unit(cellvol["payload"], "volume")
    if d:
        raise ValueError(d["message"])
    points, csyr, ccell, diags = fit_mod.molar_points(
        integ["payload"], blank_res["payload"] if blank_res else None,
        conc["payload"], discarded)
    if points is None:
        payload = {"kind": "model_fit", "model": op["model"],
                   "param_version": fit_mod.FIT_VERSION,
                   "status": "semantically_unsolvable",
                   "diagnostics": diags, "candidates": []}
    else:
        payload = fit_mod.fit_model(op["model"], points, ccell, csyr, v0)
        payload["diagnostics"] = payload.get("diagnostics", []) + diags
        payload["points"] = points
    payload["inputs"] = {
        "integrals": {"result_id": integ["id"],
                      "result_kind": "injection_integrals",
                      "result_seq": integ["seq"]},
        "concentrations": {"id": conc["id"], "kind": "concentrations",
                           "version": conc["version"]},
        "cell_volume": {"id": cellvol["id"], "kind": "cell_volume",
                        "version": cellvol["version"]},
        "decisions": list(decision_events),
    }
    if blank_res:
        payload["inputs"]["blank"] = {
            "result_id": blank_res["id"],
            "result_kind": "blank_subtraction",
            "result_seq": blank_res["seq"]}
    parents = [integ["id"], conc["id"], cellvol["id"]] + list(decision_events)
    if blank_res:
        parents.append(blank_res["id"])
    res_id = _result_id("model_fit", payload)
    with store.operation("fit", {"model": op["model"]}):
        evt = store.put_event("fit", {
            "model": op["model"], "status": payload["status"],
            "n_candidates": len(payload.get("candidates", [])),
            "result_id": res_id},
            parents=parents, input_digest=digest_of(payload["inputs"]),
            param_version=fit_mod.FIT_VERSION, status=payload["status"])
        res = store.put_result("model_fit", evt["id"], payload,
                               status=payload["status"])
    syms[op["as"]] = res


def _op_compare(store, syms, op):
    fits = [syms[n] for n in op["fits"]]
    payload = fit_mod.compare_models([f["payload"] for f in fits])
    payload["inputs"] = {
        "fits": [{"result_id": f["id"], "result_kind": "model_fit",
                  "result_seq": f["seq"]} for f in fits]}
    res_id = _result_id("model_comparison", payload)
    with store.operation("compare"):
        evt = store.put_event("compare", {
            "equally_good": payload["equally_good"], "result_id": res_id},
            parents=[f["id"] for f in fits],
            input_digest=digest_of(payload["inputs"]),
            param_version=fit_mod.FIT_VERSION)
        res = store.put_result("model_comparison", evt["id"], payload)
    syms[op["as"]] = res


def _op_decide(store, syms, op, discarded, decision_events):
    dec = dict(op["decision"])
    parents = []
    if "target_result" in dec:
        target = syms[dec.pop("target_result")]
        dec["target"] = target["id"]
        if "candidate" in dec:
            cands = target["payload"].get("candidates", [])
            if cands and dec["candidate"] < len(cands):
                dec["target_candidate"] = cands[dec["candidate"]]["candidate_id"]
        parents.append(target["id"])
    if dec.get("action") == "discard_injection":
        discarded.add(dec["target"])
    with store.operation("decide", {"action": dec.get("action")}):
        evt = store.put_event("decision", {"decision": dec}, parents=parents,
                              param_version="review/1.0")
    decision_events.append(evt["id"])


def _op_scope_check(store, syms, op):
    observed, passed = _eval_check(store, syms, op["check"])
    payload = {"rule_id": op["rule_id"], "statement": op["statement"],
               "expect": op["check"], "observed": observed, "passed": passed}
    with store.operation("scope_check", {"rule_id": op["rule_id"]}):
        store.put_event("scope_check", payload,
                        input_digest=digest_of(payload["expect"]),
                        param_version="scope/1.0",
                        status="ok" if passed else "failed")


def _eval_check(store, syms, check):
    """评估一条机器可读判定, 返回 (observed, passed)。"""
    t = check["type"]
    if t == "snapshot":
        snap = {}
        for kind in MATERIAL_KINDS:
            m = store.latest_material(kind)
            if m:
                snap[kind] = {"id": m["id"], "version": m["version"],
                              "digest": m["digest"]}
        return snap, len(snap) == len(MATERIAL_KINDS)
    if t == "integral_status":
        res = syms[check["result"]]
        byid = {i["injection_id"]: i for i in res["payload"]["injections"]}
        observed = {iid: byid.get(iid, {}).get("status")
                    for iid in check["injections"]}
        return observed, all(s == check["expect"] for s in observed.values())
    if t == "first_injection":
        integ = syms[check["integrals"]]
        fit_res = syms[check["fit"]]
        iid = check["injection"]
        byid = {i["injection_id"]: i for i in integ["payload"]["injections"]}
        retained = (iid in byid and byid[iid]["status"] == "ok"
                    and byid[iid]["heat_uJ"] is not None)
        excluded = all(iid not in c["injection_ids"]
                       for c in fit_res["payload"].get("candidates", []))
        return {"integral_retained": retained, "excluded_from_fit": excluded}, \
            retained and excluded
    if t == "blank_missing":
        res = syms[check["result"]]
        missing = [e["index"] for e in res["payload"]["series"]["blank_corrected"]
                   if e["status"] == "blank_missing"]
        raw_present = len(res["payload"]["series"]["raw"]) > 0
        return {"missing": missing, "raw_series_present": raw_present}, \
            sorted(missing) == sorted(check["expect"]) and raw_present
    if t == "concentration_scope":
        integ = syms[check["integrals"]]
        _, integ_reasons = views.staleness(store, integ)
        integ_ok = not any(r.startswith("concentrations") for r in integ_reasons)
        fit_reasons = {}
        for name in check["stale_fits"]:
            _, reasons = views.staleness(store, syms[name])
            fit_reasons[name] = reasons
        fits_stale_by_conc = all(
            any(r.startswith("concentrations") for r in reasons)
            for reasons in fit_reasons.values())
        return {"integrals_conc_reasons": integ_ok,
                "fit_reasons": fit_reasons}, integ_ok and fits_stale_by_conc
    if t == "equal_models":
        cmp_res = syms[check["compare"]]
        p = cmp_res["payload"]
        multi = any(m.get("n_candidates", 0) >= 2 for m in p["models"])
        eq = len(p["equally_good"]) >= 2
        ordered = all(
            m.get("n_candidates", 0) >= 1 for m in p["models"] if m["status"] == "ok")
        return {"equally_good": p["equally_good"],
                "models": p["models"],
                "has_multi_candidate": multi}, ordered and (multi or eq)
    if t == "flags":
        res = syms[check["result"]]
        byid = {i["injection_id"]: i for i in res["payload"]["injections"]}
        observed, ok = {}, True
        for iid, want in check["expect"].items():
            got = byid.get(iid, {}).get("flags", [])
            observed[iid] = got
            ok = ok and all(w in got for w in want)
        return observed, ok
    if t == "injection_in_gap":
        res = syms[check["result"]]
        byid = {i["injection_id"]: i for i in res["payload"]["injections"]}
        inj = byid[check["injection"]]
        others_ok = all(i["status"] == "ok" for i in res["payload"]["injections"]
                        if i["injection_id"] != check["injection"])
        diag_ok = any("position" in d for d in inj["diagnostics"])
        observed = {"status": inj["status"], "flags": inj["flags"],
                    "heat_uJ": inj["heat_uJ"], "others_ok": others_ok}
        return observed, (inj["status"] == "error"
                          and "injection_in_gap" in inj["flags"]
                          and inj["heat_uJ"] is None and diag_ok and others_ok)
    if t == "unchanged_results":
        observed, ok = {}, True
        for name in check["results"]:
            res = syms[name]
            again = store.result(res["id"])
            same = again is not None and again["digest"] == res["digest"]
            observed[name] = {"id": res["id"], "unchanged": same}
            ok = ok and same
        return observed, ok
    if t == "blank_rules":
        observed = [syms[n]["payload"]["rule"]["kind"] for n in check["results"]]
        return {"rules": observed}, observed == check["expect_rules"]
    if t == "result_status":
        res = syms[check["result"]]
        return {"status": res["status"]}, res["status"] == check["expect"]
    raise ValueError(f"未知核验类型 {t}")


def load_casepack(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_casepack(path):
    pack = build_casepack()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pack, fh, ensure_ascii=False)
    return path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "casepack/itc_titration_case.json"
    write_casepack(out)
    print(f"written {out}")
