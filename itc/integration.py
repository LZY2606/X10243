"""注射积分规则。

关键判断（领域规则在服务端，可直接被测试调用）：
- 注射后峰可以重叠：重叠只打标，不因此合并或拒算。
- 积分窗不跨越数据缺口：遇到缺口只保留“从窗左端开始的首段连续有效区间”，
  端点缺口（左/右端）与内部缺口分别给出带位置诊断。
- 基线是独立版本化材料：线性插值到功率时间基后相减。
- 首注舍弃是规则决定：原始积分仍保留，只是排除出后续拟合。
- 每个条目带 input_digest（内容摘要），用于验证局部源数据改动的影响范围。
"""
from __future__ import annotations

import numpy as np

from .models import Diagnostic, ITCError
from .storage import digest

INTEGRATION_RULE_VERSION = "integration-1.0.0"
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# 端点判定容差：采样点落在注射时刻后 1 个采样间隔内视为正常起点
ENDPOINT_TOL_FACTOR = 1.5


def _series(piece: dict) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    t = np.asarray(piece["time"]["value"], dtype=float)
    raw = piece["power"]["raw"]
    v = np.array([np.nan if x is None else float(x) for x in
                  piece["power"]["value"]], dtype=float)
    return t, v, raw


def gap_intervals(times: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    """缺口区间：连续缺失采样点构成的 (t_start, t_end)。"""
    missing = np.isnan(values)
    gaps = []
    i = 0
    n = len(times)
    while i < n:
        if missing[i]:
            j = i
            while j + 1 < n and missing[j + 1]:
                j += 1
            gaps.append((float(times[i]), float(times[min(j, n - 1)])))
            i = j + 1
        else:
            i += 1
    return gaps


def baseline_on_grid(baseline: dict | None, times: np.ndarray) -> np.ndarray:
    if baseline is None:
        return np.zeros_like(times)
    bt = np.asarray(baseline["time"]["value"], dtype=float)
    bv = np.asarray(baseline["power"]["value"], dtype=float)
    return np.interp(times, bt, bv)


def integrate_injections(power: dict, baseline: dict | None,
                         injections: dict, params: dict) -> dict:
    """对每次注射做基线扣除后积分。

    params:
      window_s        注射后积分窗长（秒，时间基准：功率序列首样本起的秒）
      discard_ordinals 首注舍弃的序号列表（规则决定，不删原积分）
    返回积分产物文档（含每条注射的有效区间、缺口、诊断与输入摘要）。
    """
    window_s = params.get("window_s")
    if window_s is None or float(window_s) <= 0:
        raise ITCError(
            "积分窗长 window_s 必须为正数",
            [Diagnostic("missing_window", "window_s 必填且为正", "params.window_s")],
        )
    window_s = float(window_s)
    discard = {int(x) for x in params.get("discard_ordinals", [])}

    t, p, raw_p = _series(power)
    base = baseline_on_grid(baseline, t)
    net = p - base
    sample_dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    gaps_all = gap_intervals(t, p)
    t0, t_end = float(t[0]), float(t[-1])
    schedule = injections["schedule"]

    # 先确定名义窗，做重叠判定（峰可重叠：仅标记）
    nominals = []
    for s in schedule:
        a = float(s["time_s"])
        b = min(a + window_s, t_end)
        nominals.append((s["index"], a, b))
    overlap_of: dict[int, list[int]] = {}
    for i in range(len(nominals)):
        oi, ia, ib = nominals[i]
        for j in range(len(nominals)):
            if i == j:
                continue
            oj, oa, ob = nominals[j]
            # 名义窗在实际注射时刻上相交（后一次注射落在前一次窗内）
            if ia < oa < ib - 1e-12 or oa < ia < ob - 1e-12:
                overlap_of.setdefault(oi, []).append(oj)

    entries = []
    diagnostics: list[Diagnostic] = []
    for s, (ordinal, nstart, nend) in zip(schedule, nominals):
        pos = f"injection #{ordinal}"
        ediags: list[Diagnostic] = []
        if nstart > t_end:
            ediags.append(Diagnostic(
                "injection_after_data", "注射时刻晚于最后一个采样点",
                pos, severity="error"))
            entries.append(_entry(s, nstart, nend, None, None, None, 0.0,
                                  0.0, [], overlap_of.get(ordinal, []),
                                  ordinal in discard, ediags,
                                  power, baseline, injections, params, sample_dt))
            continue
        if nstart < t0:
            ediags.append(Diagnostic(
                "injection_before_data", "注射时刻早于第一个采样点",
                pos, severity="error"))

        # 落在名义窗内的采样点（包含右端）
        in_win = np.where((t >= nstart - 1e-12) & (t <= nend + 1e-12))[0]
        seg_idx = list(in_win)
        # 首段连续有效点：不跨越任何缺口
        cut_reason = None
        excluded_gaps = []
        valid_run = []
        started = False
        for k in in_win:
            if np.isnan(p[k]):
                gap_t = float(t[k])
                if not started:
                    # 左端点缺口：跳过，直到出现有效点（端点裁剪）
                    started = True if not valid_run else started
                    cut_reason = cut_reason or "left_endpoint_gap"
                    continue
                # 遇到第一个内部缺口：截断，绝不跨越
                gj = [g for g in gaps_all if abs(g[0] - gap_t) < 1e-9]
                excluded_gaps.extend(gj)
                cut_reason = "interior_gap" if valid_run else cut_reason
                break
            started = True
            valid_run.append(int(k))
        # 右端是否被缺口截断
        if valid_run:
            last_k = valid_run[-1]
            if last_k + 1 < len(t) and np.isnan(p[last_k + 1]) and \
                    t[last_k] < nend - 1e-9:
                cut_reason = cut_reason or "right_endpoint_gap"
                gj = [g for g in gaps_all
                      if abs(g[0] - float(t[last_k + 1])) < 1e-9]
                excluded_gaps.extend(gj)

        if not valid_run:
            ediags.append(Diagnostic(
                "no_valid_samples", "积分窗内没有任何有效采样点（无法积分）",
                pos, severity="error"))
            entries.append(_entry(s, nstart, nend, None, None, None, 0.0,
                                  0.0, gaps_touching(gaps_all, nstart, nend),
                                  overlap_of.get(ordinal, []),
                                  ordinal in discard, ediags,
                                  power, baseline, injections, params, sample_dt))
            continue

        ki = np.array(valid_run, dtype=int)
        seg_t = t[ki]
        seg_net = net[ki]
        eff_start, eff_end = float(seg_t[0]), float(seg_t[-1])
        q_uJ = float(_trapz(seg_net, seg_t))
        coverage = (eff_end - eff_start) / (nend - nstart) if nend > nstart else 0.0

        if cut_reason == "left_endpoint_gap":
            ediags.append(Diagnostic(
                "window_left_clipped",
                "窗左端存在缺口，积分起点裁剪到首个有效采样点",
                f"{pos} @t={eff_start:g}s", severity="warning"))
        if cut_reason in ("interior_gap", "right_endpoint_gap"):
            ediags.append(Diagnostic(
                "window_gap_clipped",
                "积分窗在数据缺口处截断（未跨越缺口）",
                f"{pos} 有效区间结束于 t={eff_end:g}s", severity="warning"))
        if coverage < 0.999 and not cut_reason:
            ediags.append(Diagnostic(
                "window_partial_coverage", "窗未被采样点完全覆盖",
                pos, severity="warning"))
        if ordinal in discard:
            ediags.append(Diagnostic(
                "first_injection_discarded",
                "规则决定舍弃该注射（原积分保留，不进入模型）",
                pos, severity="info"))
        if overlap_of.get(ordinal):
            ediags.append(Diagnostic(
                "overlapping_peak",
                "该注射的名义积分窗与其它注射窗相交（峰允许重叠，单独积分）",
                f"{pos} 与 #{','.join(map(str, overlap_of[ordinal]))} 重叠",
                severity="info"))

        entries.append(_entry(
            s, nstart, nend, eff_start, eff_end,
            [float(x) for x in seg_t], q_uJ, coverage,
            merge_gaps(excluded_gaps) or gaps_touching(gaps_all, nstart, nend),
            overlap_of.get(ordinal, []), ordinal in discard, ediags,
            power, baseline, injections, params, sample_dt,
            seg_raw=[raw_p[k] for k in ki], seg_power=[float(p[k]) for k in ki],
            seg_baseline=[float(base[k]) for k in ki]))

    diagnostics.extend(_collect(entries))
    inputs = {
        "power_material": power["id"],
        "baseline_material": baseline["id"] if baseline else None,
        "injections_material": injections["id"],
        "power_samples": len(t),
        "gap_count": len(gaps_all),
    }
    return {
        "kind": "integration",
        "rule_version": INTEGRATION_RULE_VERSION,
        "time_basis": "seconds since first power sample",
        "energy_unit": "uJ",
        "power_unit": "uW",
        "params": {"window_s": window_s, "discard_ordinals": sorted(discard)},
        "inputs": inputs,
        "entries": entries,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def _entry(s, nstart, nend, eff_start, eff_end, seg_t, q_uJ, coverage,
           gaps_touch, overlaps, discarded, ediags, power, baseline,
           injections, params, sample_dt, seg_raw=None, seg_power=None,
           seg_baseline=None):
    basis = {
        "power": power["id"], "baseline": baseline["id"] if baseline else None,
        "injections": injections["id"],
        "injection": {"index": s["index"], "time_s": s["time_s"],
                      "volume_uL": s["volume_uL"]},
        "window_s": params.get("window_s"),
        "rule": INTEGRATION_RULE_VERSION,
        "samples": list(zip(seg_t or [], seg_power or [], seg_baseline or [],
                            seg_raw or [])),
    }
    fatal = [d for d in ediags if d.severity == "error"]
    return {
        "ordinal": s["index"],
        "injection_time_s": s["time_s"],
        "injection_volume_uL": s["volume_uL"],
        "nominal_window_s": [nstart, nend],
        "effective_window_s": [eff_start, eff_end] if seg_t else None,
        "n_points": len(seg_t or []),
        "coverage": round(coverage, 9),
        "q_gross_uJ": round(q_uJ, 9) if seg_t else None,
        "excluded_gap_intervals_s": [[round(a, 6), round(b, 6)]
                                     for a, b in gaps_touch],
        "overlaps_with": sorted(overlaps),
        "discarded": discarded,
        "fatal": bool(fatal),
        "input_digest": digest(basis),
        "diagnostics": [d.to_dict() for d in ediags],
    }


def _collect(entries):
    out = []
    for e in entries:
        for d in e["diagnostics"]:
            out.append(Diagnostic.from_dict(d))
    return out


def gaps_touching(gaps, a, b):
    return [g for g in gaps if g[1] >= a - 1e-9 and g[0] <= b + 1e-9]


def merge_gaps(gs):
    out = []
    for g in gs:
        if not out or abs(g[0] - out[-1][1]) > 1e-9:
            out.append(g)
    return out
