"""注射积分引擎（派生层）。

硬性领域规则：
* 注射后峰 *可以重叠* —— 相邻注射窗口在两峰之间的谷点切分
  （overlap_rule=valley_split），切分点有坐标证据；
* 积分窗 *绝不跨越数据缺口* —— 时间跳跃或无效采样行把窗口截断在缺口前，
  并记录 clipped_by_gap 与有效区间；
* 端点可以是“回到基线”“谷点切分”“被缺口截断”“轨迹结束”，
  每一种都保留判定状态与证据点；
* 首注舍弃只是规则决定：原始积分照常计算并保留，只是被标记不进入拟合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .materials import MaterialVersion
from .units import numeric_array


@dataclass
class IntegrationRules:
    first_injection_discard: bool = True
    overlap_rule: str = "valley_split"          # 唯一支持：谷点切分
    gap_factor: float = 5.0                     # dt > gap_factor * median(dt) 视为缺口
    settle_seconds: float = 3.0                 # 连续多久低于容差才算回到基线
    baseline_tol_ucal: float = 0.2              # µcal/s
    valley_search_seconds: float = 6.0          # 在下一注前的谷点搜索窗
    max_window_seconds: float = 600.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "first_injection_discard": self.first_injection_discard,
            "overlap_rule": self.overlap_rule,
            "gap_factor": self.gap_factor,
            "settle_seconds": self.settle_seconds,
            "baseline_tol_ucal": self.baseline_tol_ucal,
            "valley_search_seconds": self.valley_search_seconds,
            "max_window_seconds": self.max_window_seconds,
        }


@dataclass
class InjectionIntegral:
    ordinal: int
    injection_time_s: float
    volume_l: float
    window_start_s: Optional[float]
    window_end_s: Optional[float]
    valid: bool
    integral_corrected_cal: Optional[float]
    integral_uncorrected_cal: Optional[float]
    peak: Optional[Dict[str, float]]
    valley: Optional[Dict[str, float]]
    baseline_return_at_s: Optional[float]
    status: List[str] = field(default_factory=list)
    diagnostics: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "injection_time_s": self.injection_time_s,
            "volume_l": self.volume_l,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "valid": self.valid,
            "integral_corrected_cal": self.integral_corrected_cal,
            "integral_uncorrected_cal": self.integral_uncorrected_cal,
            "peak": self.peak,
            "valley": self.valley,
            "baseline_return_at_s": self.baseline_return_at_s,
            "status": list(self.status),
            "diagnostics": list(self.diagnostics),
        }


def _col(mat: MaterialVersion, name: str) -> np.ndarray:
    arr, _ = numeric_array(mat.columns[name].quantities)
    return arr


def resolve_trace(power: MaterialVersion, baseline: Optional[MaterialVersion]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (t, p_raw, p_corrected, valid_mask)，全部内部 SI 单位。"""
    t = _col(power, "time")
    p = _col(power, "power")
    valid = ~np.isnan(t) & ~np.isnan(p)
    if baseline is not None:
        bt = _col(baseline, "time")
        bp = _col(baseline, "power")
        bvalid = ~np.isnan(bt) & ~np.isnan(bp)
        base_at = np.interp(t, bt[bvalid], bp[bvalid], left=np.nan, right=np.nan)
    else:
        base_at = np.zeros_like(p)
    p_corr = p - base_at
    valid = valid & ~np.isnan(p_corr)
    return t, p, p_corr, valid


def find_gaps(t: np.ndarray, valid: np.ndarray, gap_factor: float) -> List[Tuple[float, float]]:
    """返回数据缺口区间 [(gap_start_t, gap_end_t), ...]。

    包括相邻有效点之间的异常时间跳跃，以及无效采样行造成的缺口。
    """
    gaps: List[Tuple[float, float]] = []
    vt = t[valid]
    if vt.size < 2:
        return gaps
    dts = np.diff(vt)
    med = np.median(dts)
    if med <= 0:
        return gaps
    jump = dts > gap_factor * med
    for k in np.where(jump)[0]:
        gaps.append((float(vt[k]), float(vt[k + 1])))
    return gaps


def integrate_run(power: MaterialVersion, injections: MaterialVersion,
                  baseline: Optional[MaterialVersion], rules: IntegrationRules) -> Dict[str, Any]:
    """对一次运行的全部注射做积分，返回结构化结果与证据。"""
    t, p_raw, p_corr, valid = resolve_trace(power, baseline)
    gaps = find_gaps(t, valid, rules.gap_factor)

    inj_t = _col(injections, "time")
    inj_v = _col(injections, "volume")
    inj_o = _col(injections, "ordinal")
    inj_valid = ~np.isnan(inj_t) & ~np.isnan(inj_v) & ~np.isnan(inj_o)

    order = np.argsort(inj_t[inj_valid])
    tv = t[valid]
    pv = p_corr[valid]
    praw = p_raw[valid]
    trace_end = float(tv[-1]) if tv.size else np.nan
    tol = rules.baseline_tol_ucal * 1e-6  # µcal/s -> cal/s

    records: List[InjectionIntegral] = []
    raw_injs = []
    for idx in order:
        raw_injs.append((int(round(inj_o[inj_valid][idx])), float(inj_t[inj_valid][idx]),
                         float(inj_v[inj_valid][idx])))

    for n, (ordinal, t0, vol) in enumerate(raw_injs):
        next_t = raw_injs[n + 1][1] if n + 1 < len(raw_injs) else None
        rec = InjectionIntegral(ordinal, t0, vol, None, None, False, None, None, None, None, None)
        if rules.first_injection_discard and n == 0:
            rec.status.append("first_injection_discarded")

        cap = min(t0 + rules.max_window_seconds, trace_end)
        # 谷点切分候选（重叠峰）
        valley: Optional[Tuple[float, float]] = None
        if next_t is not None and next_t > t0:
            lo = max(t0, next_t - rules.valley_search_seconds)
            sel = (tv >= lo) & (tv <= next_t)
            if sel.any():
                k = int(np.argmin(np.abs(pv[sel])))
                valley = (float(tv[sel][k]), float(pv[sel][k]))
        # 峰点（注射后最大绝对偏转）
        peak_zone_end = valley[0] if valley is not None else cap
        psel = (tv >= t0) & (tv <= peak_zone_end)
        peak = None
        if psel.any():
            kp = int(np.argmax(np.abs(pv[psel])))
            peak = (float(tv[psel][kp]), float(pv[psel][kp]))
        # 回到基线候选
        br_time = None
        if peak is not None:
            search = (tv >= peak[0]) & (tv <= cap)
            ts_search = tv[search]
            ps_search = np.abs(pv[search])
            for k in range(len(ts_search)):
                tail = (ts_search >= ts_search[k]) & (ts_search <= ts_search[k] + rules.settle_seconds)
                if tail.any() and np.all(ps_search[tail] <= tol):
                    br_time = float(ts_search[k])
                    break
        # 端点选择
        candidates = [cap, trace_end]
        if br_time is not None:
            candidates.append(br_time)
        if valley is not None:
            candidates.append(valley[0])
        end = min(candidates)
        start = t0
        # 重叠判定（上一注的窗越过本注）
        if n > 0 and records[n - 1].window_end_s is not None and records[n - 1].window_end_s > t0:
            rec.status.append("overlap_prev_split_at_valley")
        if valley is not None and (br_time is None or valley[0] < br_time):
            rec.status.append("overlap_next_split_at_valley")
        if br_time is not None and end == br_time:
            rec.baseline_return_at_s = br_time
            rec.status.append("returned_to_baseline")
        if valley is not None and end == valley[0]:
            rec.valley = {"t_s": valley[0], "p_cal_s": valley[1]}
            rec.status.append("endpoint_valley_split")
        # 缺口截断：窗口不跨越缺口
        for gs, ge in gaps:
            if start < gs < end:
                rec.status.append("clipped_by_gap")
                rec.diagnostics.append({"code": "gap_in_window",
                                        "message": f"积分窗在缺口 {gs:.3f}s–{ge:.3f}s 前截断",
                                        "location": f"power:t={gs:.3f}"})
                end = gs
        if end >= trace_end - 1e-12:
            rec.status.append("clipped_end_of_trace")
        seg = (tv >= start) & (tv < end)
        rec.window_start_s = float(start)
        rec.window_end_s = float(end)
        if peak is not None:
            rec.peak = {"t_s": peak[0], "p_cal_s": peak[1]}
        if not seg.any():
            rec.valid = False
            rec.diagnostics.append({"code": "empty_window",
                                    "message": f"第 {ordinal} 注积分窗内无有效采样",
                                    "location": f"injection#{ordinal}"})
            rec.status.append("no_integral")
        else:
            rec.integral_corrected_cal = float(np.trapezoid(pv[seg], tv[seg]))
            rec.integral_uncorrected_cal = float(np.trapezoid(praw[seg], tv[seg]))
            rec.valid = True
            if "clipped_by_gap" in rec.status or "clipped_end_of_trace" in rec.status:
                rec.status.append("partial_integral")
        records.append(rec)

    return {
        "rules": rules.as_dict(),
        "gaps_s": [[a, b] for a, b in gaps],
        "trace_range_s": [float(tv[0]) if tv.size else None, trace_end if tv.size else None],
        "integrals": [r.as_dict() for r in records],
    }
