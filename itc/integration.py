"""注射积分引擎（纯函数，可直接被测试调用）。

关键规则：
1. 注射后峰允许重叠：积分照常进行，但记录 overlap_with，不做峰解卷积。
2. 积分窗不跨越数据缺口：窗内存在把窗切成两段的缺口 => blocked_gap，不给数值；
   审阅者可把端点裁剪到缺口边界，得到 clipped 结果（必须留下决定事件）。
3. 稀释热扣除必须说明对应规则：same_index / manual / none；
   空白对应缺失 => net_heat 留空并附诊断，绝不静默归零。
4. 浓度修订不进入积分输入，因此永远不改原积分结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonicalid import cid

VALID = "valid"
CLIPPED = "clipped_to_gap"
BLOCKED_GAP = "blocked_gap"
DISCARDED = "discarded"


@dataclass
class IntegrationParams:
    lag_s: float = 60.0
    width_s: float = 120.0
    gap_factor: float = 5.0
    blank_rule: str = "same_index"   # same_index | manual | none
    blank_map: dict[int, int] = field(default_factory=dict)  # sample index -> blank index
    window_overrides: dict[int, dict] = field(default_factory=dict)
    discard_indices: list[int] = field(default_factory=list)


def _series(trace_payload: dict):
    pts = trace_payload["points"]
    t = np.array([p["time"]["si_value"] for p in pts], dtype=float)
    p = np.array([pw["power"]["si_value"] for pw in pts], dtype=float)
    rows = [p0["row"] for p0 in pts]
    good = ~(np.isnan(t) | np.isnan(p))
    order = np.argsort(t[good], kind="stable")
    return t[good][order], p[good][order], [rows[i] for i in np.where(good)[0][order]]


def _gaps(t, factor: float):
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med = np.median(dt)
    idx = np.where(dt > factor * med)[0]
    return [(float(t[i]), float(t[i + 1])) for i in idx]


def _baseline_at(baseline_payload, t):
    if not baseline_payload or not baseline_payload.get("points"):
        return np.zeros_like(t)
    bt, bp, _ = _series(baseline_payload)
    return np.interp(t, bt, bp)


def _trapz_with_endpoints(t, y, start, end):
    """梯形积分，端点用线性插值落到窗边界。返回 (面积, 内部点 row 列表, 实际端值)。"""
    if len(t) < 2 or start < t[0] or end > t[-1] or end <= start:
        return None
    ys = np.interp([start, end], t, y)
    m = (t >= start) & (t <= end)
    ti = np.concatenate([[start], t[m], [end]])
    yi = np.concatenate([[ys[0]], y[m], [ys[1]]])
    inside_rows = []
    rows_by_t = dict(zip(np.round(t, 9), range(len(t))))
    return float(np.trapz(yi, ti))


def _integrate_window(t, y, start, end, rows):
    if len(t) < 2 or start < t[0] - 1e-9 or end > t[-1] + 1e-9 or end <= start:
        return None, "window_outside_support"
    ys = np.interp([start, end], t, y)
    m = (t >= start) & (t <= end)
    ti = np.concatenate([[start], t[m], [end]])
    yi = np.concatenate([[ys[0]], y[m], [ys[1]]])
    point_ids = [f"row:{rows[i]}" for i in np.where(m)[0]]
    return float(np.trapz(yi, ti)), point_ids


def _blank_injection(blank_payload):
    if not blank_payload:
        return None, [], []
    bp, bs = blank_payload.get("power"), blank_payload.get("schedule")
    if not bp or not bs:
        return None, [], []
    t, y, rows = _series(bp)
    return (t, y, rows), {b["index"]: b for b in bs["injections"]
                          if b.get("index") is not None}, bp


def integrate_all(power: dict, baseline: dict | None, schedule: dict,
                  params: IntegrationParams, blank: dict | None = None,
                  versions: dict | None = None) -> dict:
    versions = versions or {}
    t, y, rows = _series(power)
    bl = _baseline_at(baseline, t)
    yc = y - bl
    gaps = _gaps(t, params.gap_factor)
    (blank_series, blank_inj, blank_power_payload) = (None, {}, None)
    bi = _blank_injection(blank)
    if bi[0] is not None:
        blank_series, blank_inj, blank_power_payload = bi
    bt = blank_series[0] if blank_series else None
    by = blank_series[1] if blank_series else None
    brows = blank_series[2] if blank_series else None
    bbl = _baseline_at(blank_power_payload.get("baseline") if blank_power_payload else None, bt) \
        if blank_series else np.array([])
    if blank_series is not None:
        by = by - bbl

    injections = [i for i in schedule["injections"] if i.get("index") is not None
                  and i["time"]["si_value"] is not None
                  and i["volume"]["si_value"] is not None]
    injections.sort(key=lambda i: i["index"])
    records = []
    for k, inj in enumerate(injections):
        idx = inj["index"]
        t0 = inj["time"]["si_value"]
        ov = params.window_overrides.get(str(idx)) or params.window_overrides.get(idx)
        start = ov["start_si"] if ov else t0 + params.lag_s
        end = ov["end_si"] if ov else start + params.width_s
        rec: dict = {
            "index": idx,
            "injection_time_si": t0,
            "injection_volume_ul": inj["volume"]["si_value"],
            "window": {"start_si": start, "end_si": end,
                       "basis": "manual_override" if ov else "injection_time+lag+width",
                       "time_reference": "trace_time_seconds"},
            "overlap_with": [],
            "gaps_in_window": [],
            "status": VALID,
            "sample_heat_uj": None,
            "blank_heat_uj": None,
            "net_heat_uj": None,
            "blank_rule": params.blank_rule,
            "blank_correspondence": None,
            "volume_scaling": None,
            "point_ids": [],
            "diagnostics": [],
            "sources": {
                "power": versions.get("power"), "baseline": versions.get("baseline"),
                "schedule": versions.get("schedule"), "blank": versions.get("blank"),
                "concentration": versions.get("concentration"),
            },
        }
        for other in injections:
            if other["index"] == idx or other["time"]["si_value"] is None:
                continue
            if start < other["time"]["si_value"] < end:
                rec["overlap_with"].append(other["index"])
        crossed = [[a, b] for (a, b) in gaps if start < a and b < end]
        rec["gaps_in_window"] = crossed
        if crossed:
            rec["status"] = BLOCKED_GAP
            rec["diagnostics"].append({
                "code": "window_crosses_gap",
                "message": "积分窗跨越数据缺口；窗未积分，请裁剪端点到缺口边界",
                "location": f"injection:{idx}", "gap_intervals_si": crossed})
        else:
            area, pids = _integrate_window(t, yc, start, end, rows)
            if area is None:
                rec["status"] = BLOCKED_GAP
                rec["diagnostics"].append({"code": "window_outside_support",
                                           "message": "积分窗落在有效采样区间之外",
                                           "location": f"injection:{idx}"})
            else:
                rec["sample_heat_uj"] = area
                rec["point_ids"] = pids
                if ov:
                    rec["status"] = CLIPPED
                if idx in params.discard_indices:
                    rec["status"] = DISCARDED
        # ---- 空白对应 ----
        if params.blank_rule == "none":
            rec["blank_correspondence"] = {"rule": "none", "matched_blank_index": None}
            rec["diagnostics"].append({"code": "blank_not_subtracted",
                                       "message": "人工决定不扣除稀释热",
                                       "location": f"injection:{idx}"})
        elif blank_series is None:
            rec["blank_correspondence"] = {"rule": params.blank_rule,
                                           "matched_blank_index": None}
            rec["diagnostics"].append({"code": "blank_correspondence_missing",
                                       "message": "没有空白曲线，无法扣除稀释热",
                                       "location": f"injection:{idx}"})
        else:
            if params.blank_rule == "same_index":
                bidx = idx
            else:
                bidx = params.blank_map.get(str(idx), params.blank_map.get(idx))
            binj = blank_inj.get(bidx) if bidx is not None else None
            rec["blank_correspondence"] = {"rule": params.blank_rule,
                                           "matched_blank_index": bidx}
            if binj is None:
                rec["diagnostics"].append({
                    "code": "blank_correspondence_missing",
                    "message": f"空白注射 {bidx} 缺失；净热留空，禁止用 0 代替",
                    "location": f"injection:{idx}"})
            elif rec["sample_heat_uj"] is not None:
                bstart = binj["time"]["si_value"] + params.lag_s
                bend = bstart + params.width_s
                bgaps = _gaps(bt, params.gap_factor)
                bcross = any(bstart < a and b < bend for a, b in bgaps)
                if bcross:
                    rec["diagnostics"].append({"code": "blank_window_crosses_gap",
                                               "message": "空白积分窗跨越缺口，未扣除",
                                               "location": f"injection:{idx}"})
                else:
                    barea, _ = _integrate_window(bt, by, bstart, bend, brows)
                    if barea is not None:
                        scale = inj["volume"]["si_value"] / binj["volume"]["si_value"]
                        rec["blank_heat_uj"] = barea
                        rec["volume_scaling"] = {
                            "factor": scale,
                            "sample_volume_ul": inj["volume"]["si_value"],
                            "blank_volume_ul": binj["volume"]["si_value"],
                            "rule": "scale_by_injection_volume_ratio"}
                        if rec["status"] != DISCARDED:
                            rec["net_heat_uj"] = rec["sample_heat_uj"] - barea * scale
        if rec["net_heat_uj"] is None and rec["status"] == VALID and \
                rec["sample_heat_uj"] is not None:
            rec["status"] = "uncorrected_blank"
        rec["id"] = cid("int", {
            "index": idx, "window": rec["window"],
            "sample_heat_uj": rec["sample_heat_uj"],
            "blank_rule": params.blank_rule,
            "matched": rec["blank_correspondence"],
            "sources": {k: v["id"] if isinstance(v, dict) else v
                        for k, v in rec["sources"].items()}})
        records.append(rec)
    run_id = cid("intrun", {
        "params": {"lag_s": params.lag_s, "width_s": params.width_s,
                   "gap_factor": params.gap_factor, "blank_rule": params.blank_rule,
                   "blank_map": params.blank_map,
                   "window_overrides": params.window_overrides,
                   "discard_indices": sorted(params.discard_indices)},
        "record_ids": [r["id"] for r in records]})
    return {"run_id": run_id, "records": records,
            "gaps_si": gaps,
            "input_summary": {
                "n_power_samples": len(t), "trace_span_si": [float(t[0]), float(t[-1])],
                "n_injections": len(injections), "n_gaps": len(gaps),
                "lag_s": params.lag_s, "width_s": params.width_s,
                "gap_factor": params.gap_factor, "blank_rule": params.blank_rule},
            "param_version": cid("intparams", params.__dict__)}
