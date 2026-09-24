"""注射积分引擎。

核心规则：
- 注射后峰可重叠：相邻注射窗允许重叠，两峰各自完整积分并标记。
- 积分窗不跨越数据缺口：窗口在缺口起点截断并给出带位置诊断；
  起点落在缺口内的注射不可积分。
- 稀释热扣除必须说明对应规则：记录 blank 版本与逐针对应规则；
  空白对应缺失的注射保持 incomplete，不用零值伪装。
- 首注舍弃是显式决定，积分仍保留，仅排除出拟合输入。
"""

import numpy as np

from .outcomes import SEMANTIC_UNSOLVABLE, diag, failure, ok
from .units import normalize_injections, normalize_samples


def find_gaps(t, factor=3.0):
    """采样间隔显著大于中位数处视为数据缺口，返回 [(start, end), ...]。"""
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med = np.median(dt)
    gaps = []
    for i, d in enumerate(dt):
        if d > med * factor:
            gaps.append((float(t[i]), float(t[i + 1])))
    return gaps


def _interp(t, y, x):
    return float(np.interp(x, t, y))


def trapz_window(t, y, lo, hi):
    """[lo, hi] 内梯形积分，端点线性插值。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (t > lo) & (t < hi)
    ts = np.concatenate(([lo], t[mask], [hi]))
    ys = np.concatenate(([_interp(t, y, lo)], y[mask], [_interp(t, y, hi)]))
    return float(np.trapezoid(ys, ts))


def integrate_series(power_t, power_y, injections, params, gaps=None,
                     stream="power"):
    """对一条功率曲线按注射列表积分。返回 (per_injection, diagnostics, gaps)。"""
    t = np.asarray(power_t, dtype=float)
    y = np.asarray(power_y, dtype=float)
    if gaps is None:
        gaps = find_gaps(t)
    per_inj = []
    diagnostics = []
    for i, inj in enumerate(injections):
        next_t = injections[i + 1]["t"] if i + 1 < len(injections) else None
        start = inj["t"] - params.get("pre_window_s", 0.0)
        end = inj["t"] + params["post_window_s"]
        flags = []
        if next_t is not None and next_t < end:
            flags.append({"code": "overlap_with_next",
                          "message": f"与下一针 t={next_t}s 的峰重叠，两峰各自完整积分",
                          "next_t": next_t})
        valid = True
        for (gs, ge) in gaps:
            if start >= gs and start < ge:
                diagnostics.append(diag(
                    "injection_in_gap", f"{stream}.injections[{i}]",
                    f"注射 t={inj['t']}s 落在数据缺口 [{gs},{ge}]s 内，不可积分",
                    injection_index=i, gap=[gs, ge]))
                valid = False
                break
            if end > gs and start < gs:
                flags.append({"code": "window_clamped_at_gap",
                              "message": f"积分窗在数据缺口起点 {gs}s 截断，不跨越缺口",
                              "gap": [gs, ge], "clamped_end": gs})
                end = gs
        if not valid:
            per_inj.append({"index": i, "t": inj["t"], "volume_uL": inj["volume_uL"],
                            "status": "invalid", "heat_uJ": None,
                            "window": [start, end], "flags": flags})
            continue
        heat = trapz_window(t, y, start, end)
        per_inj.append({"index": i, "t": inj["t"], "volume_uL": inj["volume_uL"],
                        "status": "ok", "heat_uJ": heat,
                        "window": [start, end], "flags": flags})
    return per_inj, diagnostics, gaps


def run_integration(materials, params):
    """对当前材料版本执行一次完整积分。

    materials: {stream: {"payload":..., "version":...}}
    params: {pre_window_s, post_window_s, discard_first,
             blank_rule: {mode: "per_injection_index"|"none"}}
    返回 outcome dict；成功时 result 字段含每次注射积分。
    """
    diagnostics = []
    for stream in ("power", "injections", "baseline"):
        if stream not in materials:
            return failure(SEMANTIC_UNSOLVABLE, [diag(
                "material_missing", stream, f"缺少必需材料流: {stream}", stream=stream)])

    pt, py, d1 = normalize_samples(materials["power"]["payload"], "power")
    bt, by, d2 = normalize_samples(materials["baseline"]["payload"], "baseline")
    injections, d3 = normalize_injections(materials["injections"]["payload"])
    diagnostics += d1 + d2 + d3
    if pt is None or bt is None or injections is None:
        return failure(SEMANTIC_UNSOLVABLE, diagnostics)
    if not injections:
        return failure(SEMANTIC_UNSOLVABLE, diagnostics + [diag(
            "no_injections", "injections", "注射列表为空")])

    base_at = np.interp(np.asarray(pt), np.asarray(bt), np.asarray(by))
    y_corr = np.asarray(py) - base_at

    gaps = find_gaps(np.asarray(pt))
    per_inj, d4, gaps = integrate_series(pt, y_corr, injections, params, gaps)
    diagnostics += d4

    blank_rule = params.get("blank_rule", {"mode": "none"})
    blank_info = None
    if blank_rule.get("mode") == "per_injection_index":
        if "blank" not in materials:
            return failure(SEMANTIC_UNSOLVABLE, diagnostics + [diag(
                "material_missing", "blank",
                "稀释热扣除规则要求空白曲线，但材料流 blank 缺失", stream="blank")])
        blank_payload = materials["blank"]["payload"]
        bpt, bpy, bd1 = normalize_samples(blank_payload["power"], "blank.power")
        binj, bd2 = normalize_injections(blank_payload["injections"])
        diagnostics += bd1 + bd2
        if bpt is None or binj is None:
            return failure(SEMANTIC_UNSOLVABLE, diagnostics)
        bbase = np.interp(np.asarray(bpt), np.asarray(bt), np.asarray(by))
        blank_per, bd3, _ = integrate_series(bpt, np.asarray(bpy) - bbase, binj, params)
        diagnostics += bd3
        blank_heats = {b["index"]: b["heat_uJ"] for b in blank_per
                       if b["status"] == "ok"}
        blank_info = {"rule": "per_injection_index",
                      "blank_version": materials["blank"]["version"],
                      "description": "按注射序号逐针对应空白曲线同规则积分扣除"}
        for inj in per_inj:
            if inj["status"] != "ok":
                continue
            if inj["index"] in blank_heats:
                inj["blank_heat_uJ"] = blank_heats[inj["index"]]
                inj["net_heat_uJ"] = inj["heat_uJ"] - blank_heats[inj["index"]]
            else:
                inj["status"] = "incomplete"
                diagnostics.append(diag(
                    "blank_missing_for_injection",
                    f"blank.injections[{inj['index']}]",
                    f"空白曲线缺少第 {inj['index']} 针对应，稀释热不可扣除",
                    injection_index=inj["index"]))
    else:
        blank_info = {"rule": "none", "description": "不扣除稀释热"}
        for inj in per_inj:
            if inj["status"] == "ok":
                inj["net_heat_uJ"] = inj["heat_uJ"]

    if params.get("discard_first") and per_inj:
        per_inj[0]["discarded"] = True
        per_inj[0]["flags"].append({
            "code": "first_injection_discarded",
            "message": "首注按显式决定舍弃，积分保留但不进入拟合输入"})

    usable = [i for i in per_inj
              if i["status"] == "ok" and not i.get("discarded")]
    result = {
        "injections": per_inj,
        "gaps": [list(g) for g in gaps],
        "blank": blank_info,
        "units": {"time": "s", "power": "uJ/s", "heat": "uJ", "volume": "uL"},
        "usable_count": len(usable),
    }
    return ok(result=result, diagnostics=diagnostics)
