"""逐注射功率积分。

棘手规则（必须形成结构化状态，而不只是日志）：
- 注射后峰可以重叠：每个注射独立开窗，窗口在时间轴上允许交叠；
- 积分窗不得跨越数据缺口：窗口内出现缺口时该注射积分判为 ``blocked``，
  给出缺口位置与端点建议；
- 积分端点可人工修订（endpoints 决定），只改变该注射；
- 首注舍弃（first-discard 决定）后不产生积分点，但保留原始区间；
- 基线材料可选；无基线时以功率原值积分（诊断告知）。

能量单位：功率规范化为 µW (=µJ/s)，时间为 s，故积分 µJ；再除以 4.184 -> µcal，
本模块统一以 cal 输出（µcal * 1e-6）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .ingest import InjectionSpec, MaterialKind, MaterialRecord, Sample
from .results import Diagnostic, Outcome
from .units import FACTORS

MW_TO_CAL_PER_S = 1e-6 / 4.184          # µW(=µJ/s) -> cal/s
TAIL_FACTOR = 4.0                        # 默认窗尾 = 下一注射前（或峰衰减 4τ）
OVERLAP_FRACTION = 0.5                   # 峰值下降到该比例以下视为回落


class InjectionStatus(str, Enum):
    INTEGRATED = "integrated"
    DISCARDED = "discarded"              # 首注舍弃
    BLOCKED_GAP = "blocked_gap"          # 积分窗跨越数据缺口
    INCOMPLETE_TAIL = "incomplete_tail"  # 窗尾超出功率记录
    NO_BASELINE = "no_baseline"          # 警告级：无基线材料


@dataclass(frozen=True)
class EndpointOverride:
    """一次人工积分端点决定（可修改决定层）。"""

    injection_index: int
    start_s: float | None = None
    end_s: float | None = None
    reason: str = ""


@dataclass
class InjectionResult:
    injection_index: int
    status: InjectionStatus
    start_s: float | None
    end_s: float | None
    heat_cal: float | None = None        # 已减基线（若有）
    raw_heat_cal: float | None = None
    peak_time_s: float | None = None
    overlaps_with: tuple[int, ...] = ()
    gap_locations_s: tuple[float, ...] = ()
    diagnostics: list[Diagnostic] = field(default_factory=list)
    baseline_material_id: str | None = None
    power_material_id: str = ""
    injections_material_id: str = ""
    endpoint_decision_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "injection_index": self.injection_index,
            "status": self.status.value,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "heat_cal": self.heat_cal,
            "raw_heat_cal": self.raw_heat_cal,
            "peak_time_s": self.peak_time_s,
            "overlaps_with": list(self.overlaps_with),
            "gap_locations_s": list(self.gap_locations_s),
            "diagnostics": [vars(d) for d in self.diagnostics],
            "baseline_material_id": self.baseline_material_id,
            "power_material_id": self.power_material_id,
            "injections_material_id": self.injections_material_id,
            "endpoint_decision_id": self.endpoint_decision_id,
        }


def integration_id(power_id: str, injections_id: str, baseline_id: str | None,
                   discarded: tuple[int, ...], endpoints: tuple[EndpointOverride, ...],
                   tail_factor: float) -> str:
    body = repr((power_id, injections_id, baseline_id, sorted(discarded),
                 sorted((e.injection_index, e.start_s, e.end_s) for e in endpoints),
                 round(tail_factor, 6)))
    return "integ_" + hashlib.sha256(body.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------

def _curve_arrays(curve: MaterialRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (t_s, power_cal_per_s, is_gap)，按时间升序；缺口 power=0。"""
    pts = [(s.t.canonical_value, None if s.value is None else s.value.canonical_value)
           for s in curve.series if s.t is not None]
    pts.sort(key=lambda p: p[0])
    t = np.array([p[0] for p in pts], dtype=float)
    gap = np.array([p[1] is None for p in pts], dtype=bool)
    p = np.array([0.0 if v is None else v for _, v in pts], dtype=float)
    # µW -> cal/s
    p_cal = p * MW_TO_CAL_PER_S
    return t, p_cal, gap


def _baseline_at(baseline: MaterialRecord | None, t: np.ndarray) -> tuple[np.ndarray, bool]:
    if baseline is None or len(baseline.series) < 2:
        return np.zeros_like(t), False
    bt, bp, _ = _curve_arrays(baseline)
    interp = np.interp(t, bt, bp, left=bp[0], right=bp[-1])
    return interp, True


def _find_peak(t: np.ndarray, p: np.ndarray, lo: int) -> tuple[int, float]:
    segment_end = min(len(t), max(lo + 4, int(lo + (len(t) - lo) * 0.5)))
    rel = p[lo:segment_end] - np.min(p[lo:segment_end])
    if rel.size == 0:
        return lo, t[lo]
    k = int(np.argmax(rel))
    return lo + k, float(t[lo + k])


def _default_window(idx: int, times: list[float], peak_t: float, t: np.ndarray,
                    p: np.ndarray, tail_factor: float) -> tuple[float, float]:
    start = times[idx]
    next_t = times[idx + 1] if idx + 1 < len(times) else None
    # 在峰后用幅度衰减估计回落点
    after = t >= peak_t
    if after.any():
        ta = t[after]
        pa = p[after]
        base_level = pa[-1]
        amp = pa[0] - base_level
        if amp > 0:
            target = base_level + amp * (1 - OVERLAP_FRACTION)
            below = np.where(pa <= target)[0]
            decay_t = float(ta[below[0]]) if below.size else float(ta[-1])
        else:
            decay_t = float(ta[-1])
    else:
        decay_t = float(t[-1])
    end = min(next_t, decay_t) if next_t is not None else decay_t
    if end <= start:
        end = float(t[-1])
    return start, end


def integrate(
    *,
    power: MaterialRecord,
    injections: MaterialRecord,
    baseline: MaterialRecord | None = None,
    discarded_first_n: int = 1,
    endpoints: dict[int, EndpointOverride] | None = None,
    tail_factor: float = TAIL_FACTOR,
) -> Outcome[list[InjectionResult]]:
    """逐注射积分；返回每个注射的结构化状态。"""
    endpoints = endpoints or {}
    if power.kind is not MaterialKind.POWER:
        return Outcome.format_error("WRONG_KIND", "积分需要功率材料", "")
    if injections.kind is not MaterialKind.INJECTIONS:
        return Outcome.format_error("WRONG_KIND", "积分需要注射时序材料", "")
    valid_inj: list[InjectionSpec] = [i for i in injections.injections if i.time and i.volume]
    if len(valid_inj) != len(injections.injections):
        return Outcome.format_error("BAD_INJECTIONS", "存在时间或体积缺失的注射条目",
                                    "injections")
    t, p_cal, gap = _curve_arrays(power)
    base_line, has_base = _baseline_at(baseline, t)
    corr = p_cal - base_line
    times = [i.time.canonical_value for i in valid_inj]
    results: list[InjectionResult] = []
    peak_times: dict[int, float] = {}
    for idx, spec in enumerate(valid_inj):
        t0 = times[idx]
        lo = int(np.searchsorted(t, t0, side="left"))
        peak_k, peak_t = _find_peak(t, p_cal, min(lo, len(t) - 1))
        peak_times[idx] = peak_t
        start, end = _default_window(idx, times, peak_t, t, p_cal, tail_factor)
        ov = endpoints.get(spec.index)
        if ov is not None:
            if ov.start_s is not None:
                start = ov.start_s
            if ov.end_s is not None:
                end = ov.end_s
        res = InjectionResult(
            injection_index=spec.index,
            status=InjectionStatus.INTEGRATED,
            start_s=start, end_s=end,
            peak_time_s=peak_t,
            baseline_material_id=baseline.material_id if baseline else None,
            power_material_id=power.material_id,
            injections_material_id=injections.material_id,
            endpoint_decision_id=("ep:" + hashlib.sha256(
                repr((spec.index, ov.start_s, ov.end_s, ov.reason)).encode()
            ).hexdigest()[:12]) if ov else None,
        )
        if spec.index <= discarded_first_n:
            res.status = InjectionStatus.DISCARDED
            res.diagnostics.append(Diagnostic(
                "FIRST_INJECTION_DISCARDED",
                f"第 {spec.index} 注按首注舍弃规则不参与拟合（保留原始区间）",
                f"injection#{spec.index}", severity="warning"))
            results.append(res)
            continue
        if not has_base:
            res.diagnostics.append(Diagnostic(
                "NO_BASELINE", "未提供基线材料，按功率原值积分",
                f"injection#{spec.index}", severity="warning"))
        # 窗尾超出记录
        in_win = (t >= start) & (t <= end)
        if end > float(t[-1]) + 1e-9:
            res.status = InjectionStatus.INCOMPLETE_TAIL
            res.diagnostics.append(Diagnostic(
                "INCOMPLETE_TAIL", f"积分窗尾 {end:.3f}s 超出功率记录末端 {t[-1]:.3f}s",
                f"injection#{spec.index}"))
        # 数据缺口：积分窗不得跨越
        gap_idx = np.where(in_win & gap)[0]
        if gap_idx.size:
            res.status = InjectionStatus.BLOCKED_GAP
            res.gap_locations_s = tuple(float(x) for x in t[gap_idx])
            res.diagnostics.append(Diagnostic(
                "WINDOW_CROSSES_GAP",
                f"积分窗 [{start:.3f}, {end:.3f}]s 跨越 {gap_idx.size} 个数据缺口；"
                f"可用端点建议：止于 {t[gap_idx[0] - 1]:.3f}s",
                f"injection#{spec.index}",
                ",".join(f"{x:.3f}" for x in t[gap_idx][:5])))
            results.append(res)
            continue
        if int(in_win.sum()) < 2:
            res.status = InjectionStatus.INCOMPLETE_TAIL
            res.diagnostics.append(Diagnostic(
                "WINDOW_TOO_SHORT", "积分窗内有效采样点不足 2 个",
                f"injection#{spec.index}"))
            results.append(res)
            continue
        tw = t[in_win]
        pw = corr[in_win]
        raw = p_cal[in_win]
        res.heat_cal = float(np.trapezoid(pw, tw))
        res.raw_heat_cal = float(np.trapezoid(raw, tw))
        results.append(res)
    # 峰重叠检测：窗口在时间轴上交叠（允许且标注）
    for a in range(len(results)):
        ra = results[a]
        if ra.peak_time_s is None:
            continue
        overlaps = []
        for b in range(len(results)):
            if a == b:
                continue
            rb = results[b]
            if rb.start_s is None or rb.end_s is None:
                continue
            if ra.start_s is not None and ra.end_s is not None and \
                    ra.start_s < rb.end_s and rb.start_s < ra.end_s:
                overlaps.append(rb.injection_index)
        if overlaps:
            ra.overlaps_with = tuple(sorted(overlaps))
            ra.diagnostics.append(Diagnostic(
                "OVERLAPPING_PEAKS",
                f"注射峰与第 {','.join(map(str, overlaps))} 注窗口重叠；各窗独立积分",
                f"injection#{ra.injection_index}", severity="warning"))
    return Outcome.success(results)
