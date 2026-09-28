"""注射积分：功率曲线 -> 每次注射的热量。

核心规则（RULE_VERSION 随规则改变而升版）：
- 注射后峰允许重叠：积分窗按相邻注射时间切分，不强制峰彼此独立。
- 积分窗绝不跨越数据缺口：被缺口切断时 crosses_gap，能量为 None。
- 首注舍弃：可配置（默认舍弃首注），仅标记，不改原始功率。
- 积分端点检验：窗口两端必须落在实测样本上，容差半采样。
- 基线扣除：常数基线或版本化基线曲线。
所有结果记录输入摘要、参数版本与父依赖版本。
"""
from __future__ import annotations

import bisect

from .model import IntegralPoint

RULE_VERSION = "integral-2026.1"
HALF_SAMPLE_TOL = 0.5


def _interp(t, times, values):
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or i >= len(times) - 1:
        return None
    t0, t1 = times[i], times[i + 1]
    if t1 == t0:
        return values[i]
    return values[i] + (values[i + 1] - values[i]) * (t - t0) / (t1 - t0)


def _sample_dt(times):
    if len(times) < 2:
        return 1.0
    return min(abs(times[i + 1] - times[i])
               for i in range(min(len(times) - 1, 16)))


def _endpoint_ok(t, times, gap_mask, tol):
    for ts, g in zip(times, gap_mask):
        if not g and abs(ts - t) <= tol:
            return True
    return False


def _gap_segments_in(i0, i1, times, gap_mask):
    """把 [i0,i1] 内连续缺口合并为 [[start,end],...]，端点取相邻实测时刻。"""
    segs = []
    i = i0
    while i <= i1:
        if gap_mask[i]:
            start = times[i - 1] if i > i0 else times[i]
            j = i
            while j + 1 <= i1 and gap_mask[j + 1]:
                j += 1
            end = times[j + 1] if j + 1 <= i1 else times[j]
            segs.append([start, end])
            i = j + 1
        else:
            i += 1
    return segs


def _trapz_window(t_start, t_end, times, values, gap_mask):
    i0 = bisect.bisect_left(times, t_start)
    i1 = bisect.bisect_right(times, t_end) - 1
    tol = _sample_dt(times) * HALF_SAMPLE_TOL + 1e-9
    if i0 >= len(times) or i1 < 0 or i1 <= i0:
        return None, [[t_start, t_end]], 0, False
    start_ok = _endpoint_ok(t_start, times, gap_mask, tol)
    end_ok = _endpoint_ok(t_end, times, gap_mask, tol)
    if not (start_ok and end_ok):
        return None, [[t_start, t_end]], 0, False
    for i in range(i0, i1 + 1):
        if gap_mask[i]:
            return None, _gap_segments_in(i0, i1, times, gap_mask), 0, True
    acc = 0.0
    for i in range(i0 + 1, i1 + 1):
        acc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return acc, [], i1 - i0 + 1, True


def _baseline_at(t, baseline_data):
    if baseline_data is None:
        return 0.0
    if baseline_data.get("constant") is not None:
        return float(baseline_data["constant"])
    samples = baseline_data["samples"]
    times = [s["t"] for s in samples]
    vals = [s["p"] for s in samples if not s["gap"]]
    times_v = [s["t"] for s in samples if not s["gap"]]
    v = _interp(t, times_v, vals)
    return 0.0 if v is None else v


def integrate(power_material, protocol_injections, baseline_material,
              discard_first=True, rule_version=RULE_VERSION):
    samples = power_material.data["samples"]
    times = [s["t"] for s in samples]
    gap_mask = [bool(s["gap"]) or s["p"] is None for s in samples]
    values = [(0.0 if g else s["p"]) for s, g in zip(samples, gap_mask)]

    inj_sorted = sorted(protocol_injections, key=lambda x: x["time"])
    first_idx = inj_sorted[0]["index"] if inj_sorted else None
    points = []
    for k, inj in enumerate(inj_sorted):
        idx = inj["index"]
        win_start = inj["time"]
        win_end = inj_sorted[k + 1]["time"] if k + 1 < len(inj_sorted) \
            else times[-1]
        win_end = min(win_end, times[-1])

        overlap = [o["index"] for o in inj_sorted
                   if o["index"] != idx
                   and win_start < o["time"] < win_end]

        discarded = discard_first and idx == first_idx
        status, reason, energy, gap_segments, used, endpoint_ok = (
            "ok", "", None, [], 0, True)
        base_energy = None

        if discarded:
            status, reason = "discarded", "首注舍弃规则：首注常含装样伪影"
        else:
            energy, gap_segments, used, endpoint_ok = _trapz_window(
                win_start, win_end, times, values, gap_mask)
            if not endpoint_ok:
                status, reason = ("endpoint_failed",
                                  "积分端点未落在实测样本上")
                energy = None
            elif energy is None:
                status, reason = ("crosses_gap",
                                  "积分窗跨越数据缺口；积分窗不允许跨越缺口")
            else:
                bdata = baseline_material.data if baseline_material else None
                b0 = _baseline_at(win_start, bdata)
                b1 = _baseline_at(win_end, bdata)
                base_energy = 0.5 * (b0 + b1) * (win_end - win_start)
                energy -= base_energy

        points.append(IntegralPoint(
            injection_index=idx,
            id=f"integral:{power_material.mid()}:inj{idx}",
            win_start=win_start, win_end=win_end, energy_j=energy,
            baseline_used={
                "kind": "curve" if baseline_material else "zero",
                "version": baseline_material.mid() if baseline_material else None,
                "energy_j": base_energy,
            },
            overlap_with=sorted(overlap),
            crosses_gap=status == "crosses_gap",
            gap_segments=gap_segments,
            endpoints=[win_start, win_end], endpoint_ok=endpoint_ok,
            discarded=discarded, status=status, reason=reason,
            inputs={
                "power": power_material.mid(),
                "baseline": baseline_material.mid() if baseline_material else None,
                "rule_version": rule_version,
                "window": [win_start, win_end],
                "sample_dt": _sample_dt(times),
            },
            samples_used=used))
    points.sort(key=lambda p: p.injection_index)
    return points
