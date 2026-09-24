"""注射积分引擎。

机器可读的判定规则（rule_basis 随结果与事件一起保存）：
- 注射后峰可重叠：下一针可在前一针回到基线前开始，积分窗相邻相接。
- 积分窗不跨越数据缺口：窗口在缺口起点处截断并标记，绝不跨缺口积分。
- 首注舍弃是显式选项，被舍弃的针保留记录而非删除。
- 基线缺失时返回 semantic_unresolved，不用零基线伪装成功。
- 资源预算耗尽返回 budget_exhausted；输入结构损坏返回 format_error。
"""

from __future__ import annotations

import numpy as np

DEFAULT_OPTS = {
    "discard_first": False,
    "max_window_s": 240.0,
    "min_window_s": 20.0,
    "overlap_ratio": 0.05,
    "gap_factor": 3.0,
    "budget": 2_000_000,
}

RULE_BASIS = {
    "rule": "integration_window_policy",
    "peaks_may_overlap": True,
    "window_must_not_cross_data_gap": True,
    "first_injection_discard_is_explicit": True,
    "meaning": "注射后峰可重叠，积分窗不跨越数据缺口；首注舍弃需显式声明",
}


def find_gaps(t: np.ndarray, gap_factor: float) -> list:
    """按采样间隔中位数的倍数识别数据缺口。"""
    if len(t) < 2:
        return []
    dt = np.diff(t)
    med = float(np.median(dt))
    idx = np.where(dt > med * gap_factor)[0]
    return [[float(t[i]), float(t[i + 1])] for i in idx]


def integrate(trace: dict, injections: list, baseline: dict | None,
              opts: dict | None = None) -> dict:
    """对每次注射积分。trace/baseline 为 units.normalize_trace 的派生输出。"""
    o = dict(DEFAULT_OPTS, **(opts or {}))
    if "format_error" in trace or (baseline and "format_error" in baseline):
        bad = trace.get("format_error") or baseline.get("format_error")
        return {"status": "format_error", "injections": [],
                "diagnostics": [{"path": "trace", "code": "format_error",
                                 "message": bad}],
                "rule_basis": RULE_BASIS}
    if baseline is None:
        return {"status": "semantic_unresolved", "injections": [],
                "diagnostics": [{"path": "baseline", "code": "baseline_missing",
                                 "message": "缺少基线曲线，禁止用零基线代替"}],
                "rule_basis": RULE_BASIS}
    t = np.asarray(trace["t"], dtype=float)
    y = np.asarray(trace["y"], dtype=float)
    tb = np.asarray(baseline["t"], dtype=float)
    yb = np.asarray(baseline["y"], dtype=float)
    if len(t) < 2 or len(tb) < 2:
        return {"status": "format_error", "injections": [],
                "diagnostics": [{"path": "trace", "code": "format_error",
                                 "message": "曲线有效采样点不足"}],
                "rule_basis": RULE_BASIS}
    gaps = find_gaps(t, o["gap_factor"])
    base_at = np.interp(t, tb, yb)
    budget = o["budget"]
    results, diags = [], list(trace.get("diagnostics", []))
    times = [inj["t"] for inj in injections if inj.get("ok") and inj["t"] is not None]
    for pos, inj in enumerate(injections):
        entry = {"index": inj["index"], "t": inj["t"],
                 "volume_uL": inj.get("volume_uL"), "window": None,
                 "heat_uJ": None, "flags": [], "status": "ok",
                 "diagnostics": []}
        if o["discard_first"] and pos == 0:
            entry["flags"].append("discarded_first")
            entry["status"] = "excluded"
            results.append(entry)
            continue
        if not inj.get("ok") or inj["t"] is None or inj.get("volume_uL") is None:
            entry["status"] = "unresolved"
            entry["diagnostics"].append(
                {"path": f"injections[{inj['index']}]", "code": "input_incomplete",
                 "message": "注射时间或体积缺失/不可转换"})
            results.append(entry)
            continue
        start = float(inj["t"])
        later = [x for x in times if x > start]
        end = min(later) if later else start + o["max_window_s"]
        end = min(end, start + o["max_window_s"])
        in_gap = any(g0 <= start <= g1 for g0, g1 in gaps)
        if in_gap:
            entry["status"] = "unresolved"
            entry["diagnostics"].append(
                {"path": f"injections[{inj['index']}]", "code": "injection_in_data_gap",
                 "message": f"注射时刻 {start}s 落在数据缺口内"})
            results.append(entry)
            continue
        for g0, g1 in gaps:
            if start < g0 < end:
                end = g0  # 积分窗不跨越数据缺口：在缺口起点截断
                entry["flags"].append("window_truncated_by_gap")
        end = min(end, float(t[-1]))  # 积分端点不越出曲线末端
        if end - start < o["min_window_s"]:
            entry["flags"].append("window_below_min")
            entry["window"] = [start, float(end)]
            entry["status"] = "unresolved"
            entry["diagnostics"].append(
                {"path": f"injections[{inj['index']}]", "code": "window_below_min",
                 "message": "积分窗被缺口/端点截断到过短，热量不可靠"})
            results.append(entry)
            continue
        mask = (t >= start) & (t <= end)
        budget -= int(mask.sum())
        if budget < 0:
            return {"status": "budget_exhausted", "injections": results,
                    "diagnostics": diags + [{
                        "path": f"injections[{inj['index']}]",
                        "code": "budget_exhausted",
                        "message": "积分资源预算耗尽"}],
                    "rule_basis": RULE_BASIS}
        if mask.sum() < 2:
            entry["status"] = "unresolved"
            entry["diagnostics"].append(
                {"path": f"injections[{inj['index']}]", "code": "window_empty",
                 "message": "积分窗内有效采样点不足"})
            results.append(entry)
            continue
        dev = y[mask] - base_at[mask]
        heat = float(np.trapezoid(dev, t[mask]))
        peak = float(np.max(np.abs(dev)))
        end_dev = float(abs(dev[-1]))
        overlapped = (later and abs(end - min(later)) < 1e-9
                      and peak > 0 and end_dev > o["overlap_ratio"] * peak)
        if overlapped:
            entry["flags"].append("overlapped_peak")
        entry["window"] = [start, float(end)]
        entry["heat_uJ"] = heat
        results.append(entry)
    return {"status": "ok", "injections": results, "gaps": gaps,
            "diagnostics": diags, "rule_basis": RULE_BASIS,
            "basis_t": trace.get("basis_t", "epoch:unknown")}


def subtract_blank(sample: dict, blank: dict, rule: dict) -> dict:
    """稀释热扣除。对应规则必须显式给出并随结果保存。

    空白对应缺失的针保持未扣除状态并附诊断（semantic_unresolved），
    不得用零或邻近值顶替。
    """
    basis = {"rule": "dilution_heat_subtraction",
             "matching": rule.get("matching"),
             "meaning": "稀释热扣除必须说明对应规则；"
                        "空白对应缺失的针保持未扣除并附诊断"}
    if sample.get("status") != "ok" or blank.get("status") != "ok":
        return {"status": "semantic_unresolved", "heats": [],
                "diagnostics": [{"path": "blank", "code": "upstream_unresolved",
                                 "message": "上游积分未成功，无法扣除"}],
                "rule_basis": basis}
    matching = rule.get("matching", "index")
    blank_by_index = {b["index"]: b for b in blank["injections"]}
    blank_by_vol = {}
    for b in blank["injections"]:
        if b.get("volume_uL") is not None:
            blank_by_vol.setdefault(round(b["volume_uL"], 6), b)
    heats, unresolved = [], 0
    for s in sample["injections"]:
        if s["status"] == "excluded":
            heats.append({"index": s["index"], "net_uJ": None,
                          "status": "excluded", "flags": s["flags"]})
            continue
        if matching == "index":
            b = blank_by_index.get(s["index"])
        else:
            b = blank_by_vol.get(round(s["volume_uL"] or -1, 6))
        if b is None or b.get("heat_uJ") is None:
            unresolved += 1
            heats.append({"index": s["index"], "net_uJ": None,
                          "status": "unresolved", "flags": s["flags"],
                          "diagnostics": [{
                              "path": f"injections[{s['index']}]",
                              "code": "blank_correspondence_missing",
                              "message": f"按 {matching} 规则找不到对应空白针"}]})
            continue
        if s["status"] != "ok" or s["heat_uJ"] is None:
            heats.append({"index": s["index"], "net_uJ": None,
                          "status": "unresolved", "flags": s["flags"],
                          "diagnostics": s.get("diagnostics", [])})
            unresolved += 1
            continue
        entry = {"index": s["index"], "net_uJ": s["heat_uJ"] - b["heat_uJ"],
                 "raw_uJ": s["heat_uJ"], "blank_uJ": b["heat_uJ"],
                 "status": "ok", "flags": s["flags"]}
        heats.append(entry)
    return {"status": "ok" if unresolved == 0 else "semantic_unresolved",
            "heats": heats, "rule_basis": basis}
