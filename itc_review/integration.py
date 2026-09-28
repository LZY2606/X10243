"""每次注射的积分（纯业务规则，服务端/存储层外可直接测试）。

关键判断：
- 注射后峰可以重叠：重叠只打 OVERLAP 标记并记录与谁重叠，不阻止积分
- 积分窗绝不跨越数据缺口：窗在缺口边缘截断，打 GAP_CLIPPED
- 独立基线曲线优先；无基线曲线时用注射前均值并显式注明
- 首注仍积分但按规则标记 DISCARDED_FIRST，模型层排除
- 端点（数据起止、缺口）被检验并显式标记
"""
from __future__ import annotations

import numpy as np

from .errors import Diagnostic, SemanticUnresolvable
from .ids import digest, stable_deps
from .sampling import Segments, build_segments

UCAL_PER_UJ = 1.0 / 4.184  # µW·s = µJ -> µcal


def _baseline_curve(baseline_payload: dict) -> tuple[np.ndarray, np.ndarray, str] | None:
    if baseline_payload is None:
        return None
    pts = [(s["time"]["canonical"], s["power"]["canonical"])
           for s in baseline_payload["samples"]
           if s["time"]["valid"] and s["power"]["valid"]]
    if len(pts) < 2:
        raise SemanticUnresolvable(
            "基线曲线有效点不足 2 个，无法插值",
            [Diagnostic("BASELINE_TOO_SHORT", baseline_payload.get("source", "baseline"),
                        f"valid_points={len(pts)}")],
        )
    bt = np.array([p[0] for p in pts])
    bp = np.array([p[1] for p in pts])
    return bt, bp, baseline_payload["id"]


def integrate_all(power: dict, schedule: dict, rules: dict,
                  baseline: dict | None = None, deps_extra: dict | None = None) -> dict:
    rv = rules["values"]
    seg = build_segments(power, rv["gap_factor"])
    bl = _baseline_curve(baseline)
    data_t0 = float(seg.times[0]) if len(seg.times) else None
    data_t1 = float(seg.times[-1]) if len(seg.times) else None

    # 名义窗口用于重叠判定
    nominal: list[tuple[float, float] | None] = []
    for inj in schedule["injections"]:
        if inj["time"]["valid"]:
            t = inj["time"]["canonical"]
            nominal.append((t - rv["pre_window_s"], t + rv["post_window_s"]))
        else:
            nominal.append(None)

    results: list[dict] = []
    for pos, inj in enumerate(schedule["injections"]):
        index = inj["index"]
        flags: list[str] = []
        diagnostics: list[dict] = []
        overlap_with: list[int] = []
        window = None
        q_ucal = None

        if not inj["time"]["valid"] or not inj["volume"]["valid"]:
            for d in inj["time"].get("diagnostics", []) + inj["volume"].get("diagnostics", []):
                diagnostics.append(d if isinstance(d, dict) else d.to_dict())
            results.append(_result(
                index, power, schedule, rules, baseline, deps_extra,
                valid=False, window=None, q_ucal=None,
                flags=["INVALID_INJECTION"], diagnostics=diagnostics,
                overlap_with=[], seg=seg,
            ))
            continue

        t = inj["time"]["canonical"]
        v_ul = inj["volume"]["canonical"]
        w0 = t - rv["pre_window_s"]
        w1 = t + rv["post_window_s"]

        # 重叠注射峰：只标记
        for j, other in enumerate(nominal):
            if other is None or j == pos:
                continue
            if other[0] < w1 and w0 < other[1]:
                overlap_with.append(schedule["injections"][j]["index"])
        if overlap_with:
            flags.append("OVERLAP")

        # 积分窗不跨越数据缺口：定位包含 t 的有效段
        located = _locate_segment(seg, t)
        if located is None:
            flags.append("NO_SEGMENT")
            diagnostics.append(Diagnostic(
                "INJECTION_IN_GAP_OR_OUTSIDE",
                f"{schedule['source']}#inj{index}",
                f"注射时刻 {t}s 不落在任何有效采样段（缺口内或采样范围外），积分窗无处安放",
                t,
            ).to_dict())
            results.append(_result(
                index, power, schedule, rules, baseline, deps_extra,
                valid=False, window={"nominal_start": w0, "nominal_end": w1},
                q_ucal=None, flags=flags, diagnostics=diagnostics,
                overlap_with=overlap_with, seg=seg, t=t,
            ))
            continue

        a, b = located
        seg_t0, seg_t1 = float(seg.times[a]), float(seg.times[b])
        c0, c1 = max(w0, seg_t0), min(w1, seg_t1)

        if data_t0 is not None and w0 < data_t0:
            flags.append("ENDPOINT_BEFORE_DATA")
        if data_t1 is not None and w1 > data_t1:
            flags.append("ENDPOINT_AFTER_DATA")
        if w0 < seg_t0 or w1 > seg_t1:
            # 被内部缺口截断（区别于数据起止端点）
            if w0 < seg_t0 and (data_t0 is None or w0 >= data_t0):
                flags.append("GAP_CLIPPED_LEFT")
            if w1 > seg_t1 and (data_t1 is None or w1 <= data_t1):
                flags.append("GAP_CLIPPED_RIGHT")
        # 端点检验：恰好落在采样点上 vs 落在采样点之间（插值）
        endpoint_on_sample = _on_sample(seg.times, c0) and _on_sample(seg.times, c1)
        if not endpoint_on_sample:
            flags.append("ENDPOINT_INTERPOLATED")

        sel = (seg.times >= c0 - 1e-9) & (seg.times <= c1 + 1e-9)
        wt = seg.times[sel]
        wp = seg.powers[sel]
        pre_sel = (wt <= t + 1e-9)
        post_sel = (wt >= t - 1e-9)

        if bl is not None:
            bt, bp, _bid = bl
            base = np.interp(wt, bt, bp)
            baseline_mode = "curve"
        else:
            pre_vals = wp[pre_sel]
            if len(pre_vals) < 2:
                flags.append("INSUFFICIENT_PRE_POINTS")
            base_level = float(np.mean(pre_vals)) if len(pre_vals) else 0.0
            base = np.full_like(wt, base_level)
            baseline_mode = "pre_mean"

        residual = wp - base
        usable_post = float(min(w1, seg_t1) - t)
        if usable_post < 0.5 * rv["post_window_s"]:
            flags.append("TRUNCATED_POST_WINDOW")
        if len(wt) < 2:
            flags.append("INSUFFICIENT_POINTS")
            q_uj = 0.0
        else:
            q_uj = float(np.trapezoid(residual, wt))
        q_ucal = q_uj * UCAL_PER_UJ

        window = {
            "nominal_start": w0,
            "nominal_end": w1,
            "used_start": float(wt[0]) if len(wt) else c0,
            "used_end": float(wt[-1]) if len(wt) else c1,
            "segment_start": seg_t0,
            "segment_end": seg_t1,
            "injection_time": t,
            "pre_points": int(pre_sel.sum()),
            "post_points": int(post_sel.sum()),
            "endpoint_on_sample": endpoint_on_sample,
            "time_unit": "s",
        }
        results.append(_result(
            index, power, schedule, rules, baseline, deps_extra,
            valid=True, window=window, q_ucal=q_ucal,
            volume_ul=v_ul, flags=flags, diagnostics=diagnostics,
            overlap_with=overlap_with, seg=seg, t=t,
            baseline_mode=baseline_mode,
            baseline_id=(bl[2] if bl else None),
        ))

    payload = {
        "kind": "integral_set",
        "results": results,
        "n_gaps": len(seg.gap_edges),
        "gap_edges": [list(g) for g in seg.gap_edges],
        "median_dt": seg.median_dt,
        "data_range": [data_t0, data_t1],
        "invalid_sample_rows": seg.invalid_points,
        "unit": {"time": "s", "power": "µW", "volume": "µL", "heat": "µcal"},
        "events": _judgement_events(results, rules),
    }
    return payload


def _judgement_events(results: list[dict], rules: dict) -> list[dict]:
    """机器可读的判断依据事件（结构状态，不只是日志）。"""
    events: list[dict] = []
    for r in results:
        idx = r["injection_index"]
        for flag in r["flags"]:
            if flag in ("OVERLAP", "GAP_CLIPPED_LEFT", "GAP_CLIPPED_RIGHT",
                        "ENDPOINT_BEFORE_DATA", "ENDPOINT_AFTER_DATA",
                        "DISCARDED_FIRST", "TRUNCATED_POST_WINDOW",
                        "NO_SEGMENT", "INVALID_INJECTION"):
                events.append({
                    "type": "INTEGRAL_JUDGEMENT",
                    "injection_index": idx,
                    "rule": flag,
                    "detail": {
                        "overlapping_with": r.get("overlapping_with", []),
                        "window": r.get("window"),
                    },
                    "rule_version": rules["id"],
                })
    return events


def _on_sample(times: np.ndarray, t: float) -> bool:
    if not len(times):
        return False
    idx = np.searchsorted(times, t)
    if idx < len(times) and abs(times[idx] - t) < 1e-7:
        return True
    if idx > 0 and abs(times[idx - 1] - t) < 1e-7:
        return True
    return False


def _locate_segment(seg: Segments, t: float) -> tuple[int, int] | None:
    for a, b in seg.segments:
        if seg.times[a] - 1e-9 <= t <= seg.times[b] + 1e-9:
            return a, b
    return None


def _result(index, power, schedule, rules, baseline, deps_extra, *, valid, window, q_ucal,
            flags, diagnostics, overlap_with, seg: Segments, t=None,
            volume_ul=None, baseline_mode=None, baseline_id=None) -> dict:
    discard_first = rules["values"]["discard_first"]
    if discard_first and index == 1:
        flags = flags + ["DISCARDED_FIRST"]
    deps = {
        "power": power["id"],
        "schedule": schedule["id"],
        "rules": rules["id"],
        "baseline": baseline["id"] if baseline else None,
        "injection_index": index,
    }
    if deps_extra:
        deps.update(deps_extra)
    rid = digest("integral", stable_deps(deps))
    return {
        "id": rid,
        "kind": "integral",
        "injection_index": index,
        "injection_time_s": t,
        "volume_ul": volume_ul,
        "valid": valid,
        "heat_ucal": q_ucal,
        "window": window,
        "flags": flags,
        "overlapping_with": overlap_with,
        "baseline_mode": baseline_mode,
        "baseline_id": baseline_id,
        "diagnostics": diagnostics,
        "deps": deps,
    }
