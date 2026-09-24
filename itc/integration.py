"""逐注射积分。

关键业务规则：
* 注射后的峰允许互相重叠 —— 每个注射拥有独立的积分窗，窗口可交叠，不做强制切分。
* 积分窗绝不跨越数据缺口：窗的终点被缺口边界夹断，缺口在起点之前则该注不可积分。
* 积分端点由“回到基线带并保持 keep_samples 个采样点”决定，下一注射时间是硬上界。
* 首注舍弃是显式策略，不是悄悄删除：被舍弃的注射保留记录并给出 finding。
* 基线可标量或序列；功率减去（插值后的）基线后做梯形积分。
"""
from __future__ import annotations

import math

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticUnresolved

DEFAULT_POLICY = {
    "discard_first": True,
    "baseline_band_sigma": 3.0,
    "keep_samples": 6,
    "gap_factor": 5.0,
    "max_samples": 200_000,
}


def canonical_samples(power_material: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """返回 (时间数组, 功率数组, 被跳过的原始行号)。不可解析样本留在材料里、不参与计算。"""
    times, powers, skipped = [], [], []
    for s in power_material["samples"]:
        t = s["time"]["canonical_value"]
        p = s["power"]["canonical_value"]
        if t is None or p is None:
            skipped.append(s["index"])
            continue
        times.append(t)
        powers.append(p)
    if len(times) < 2:
        raise SemanticUnresolved(
            "可积分采样点少于 2 个",
            [Diagnostic("insufficient_samples", "功率曲线有效点不足", "/power", len(times))],
        )
    order = np.argsort(times, kind="stable")
    return np.asarray(times, dtype=float)[order], np.asarray(powers, dtype=float)[order], skipped


def detect_gaps(t: np.ndarray, factor: float) -> tuple[list[dict], float]:
    """采样间隔大于 factor 倍中位间隔即数据缺口（含不可解析样本造成的空洞）。"""
    if len(t) < 2:
        return [], 0.0
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    gaps = []
    if median_dt > 0:
        for i, d in enumerate(dt):
            if d > factor * median_dt:
                gaps.append({
                    "gap_id": f"gap_{len(gaps) + 1:02d}",
                    "t_start": float(t[i]),
                    "t_end": float(t[i + 1]),
                    "duration": float(d),
                    "between_samples": [int(i), int(i + 1)],
                })
    return gaps, median_dt


def baseline_at(baseline_material: dict | None, t: np.ndarray) -> np.ndarray:
    if baseline_material is None:
        return np.zeros_like(t)
    if baseline_material.get("mode") == "scalar":
        return np.full_like(t, baseline_material["baseline"]["canonical_value"])
    bt, bp = [], []
    for s in baseline_material["samples"]:
        tv, pv = s["time"]["canonical_value"], s["power"]["canonical_value"]
        if tv is not None and pv is not None:
            bt.append(tv)
            bp.append(pv)
    if not bt:
        return np.zeros_like(t)
    return np.interp(t, np.asarray(bt), np.asarray(bp))


def _endpoint(t_seg: np.ndarray, y_seg: np.ndarray, band: float,
              keep: int, next_inj_time: float | None) -> tuple[int, float | None, str]:
    """回到 |y|<=band 并连续保持 keep 个点；否则取窗口末点（overlaps_next / clipped 标注）。"""
    inside = np.abs(y_seg) <= band
    run = 0
    for i, ok in enumerate(inside):
        run = run + 1 if ok else 0
        if run >= keep:
            return i, float(t_seg[i]), "returned"
    if next_inj_time is not None and t_seg[-1] >= next_inj_time - 1e-12:
        return len(t_seg) - 1, None, "overlaps_next"
    return len(t_seg) - 1, None, "end_of_trace"


def integrate(power_material: dict,
              injections_material: dict,
              baseline_material: dict | None,
              policy: dict | None = None) -> dict:
    pol = {**DEFAULT_POLICY, **(policy or {})}
    t, p, skipped = canonical_samples(power_material)
    if len(t) > pol["max_samples"]:
        raise BudgetExhausted(
            f"采样点数 {len(t)} 超出预算 {pol['max_samples']}",
            [Diagnostic("sample_budget", "积分采样预算耗尽", "/policy/max_samples", pol["max_samples"])],
        )
    gaps, median_dt = detect_gaps(t, pol["gap_factor"])

    injs = []
    for r in injections_material["records"]:
        tv = r["time"]["canonical_value"]
        vv = r["volume"]["canonical_value"]
        if tv is None or vv is None:
            continue
        injs.append({"index": r["index"], "t": float(tv), "volume": float(vv),
                     "raw_index": r["raw_index"], "raw_volume": r["raw_volume"]})
    injs.sort(key=lambda z: z["t"])
    if not injs:
        raise SemanticUnresolved(
            "没有可解析的注射点",
            [Diagnostic("insufficient_injections", "注射时序无有效行", "/injections", 0)],
        )

    base = baseline_at(baseline_material, t)
    y = p - base
    noise = float(np.std(y[: max(5, len(y) // 20)])) if len(y) >= 5 else 0.0
    band = max(pol["baseline_band_sigma"] * noise, 1e-12)

    injections: list[dict] = []
    findings: list[dict] = []

    for pos, inj in enumerate(injs):
        t0 = inj["t"]
        next_t = injs[pos + 1]["t"] if pos + 1 < len(injs) else None
        i0 = int(np.searchsorted(t, t0, side="left"))
        # 起点落在缺口内部：无法积分
        start_gap = next(
            (g for g in gaps if g["t_start"] < t0 < g["t_end"]), None)

        hard_end = t[-1]
        gap_block = None
        for g in gaps:
            if g["t_start"] > t0:
                if next_t is None or g["t_start"] < next_t:
                    hard_end = min(hard_end, g["t_start"])
                    gap_block = g
                break
        seg_end = t[-1] if next_t is None else min(next_t, t[-1])
        seg_end = min(seg_end, hard_end)

        mask = (t >= t0) & (t <= seg_end + 1e-12)
        t_seg, y_seg = t[mask], y[mask]

        entry: dict = {
            "injection_id": f"inj_{inj['index']:02d}",
            "index": inj["index"],
            "ordinal": pos + 1,
            "t_inject": t0,
            "volume_l": inj["volume"],
            "raw_index": inj["raw_index"],
            "raw_volume": inj["raw_volume"],
            "discarded": False,
            "integrable": True,
            "window": {"t_start": float(t0), "t_end": None,
                       "i_start": int(i0), "i_end": None},
            "heat_cal": None,
            "heat_per_volume": None,
            "endpoint_kind": None,
            "window_overlaps_next": False,
            "clipped_by_gap": False,
            "blocked_by_gap_at_start": False,
            "gap_ids": [],
            "findings": [],
        }

        def add_finding(severity: str, code: str, message: str, loc: str, **extra):
            fid = f"{entry['injection_id']}::{code}"
            f = {"finding_id": fid, "severity": severity, "code": code,
                 "message": message, "loc": loc, "injection_index": inj["index"], **extra}
            entry["findings"].append(f)
            findings.append(f)

        if pos == 0 and pol["discard_first"]:
            entry["discarded"] = True
            entry["integrable"] = False
            add_finding("info", "first_injection_discarded",
                        "首注按策略舍弃（仪器预冲/混合误差）",
                        f"/policy/discard_first")
            injections.append(entry)
            continue

        if start_gap is not None:
            entry["integrable"] = False
            entry["blocked_by_gap_at_start"] = True
            entry["gap_ids"].append(start_gap["gap_id"])
            add_finding("error", "injection_inside_gap",
                        f"注射时刻 {t0:g}s 位于数据缺口 {start_gap['gap_id']} 内",
                        f"/injections/row/{pos}/time",
                        gap_id=start_gap["gap_id"],
                        gap_interval=[start_gap["t_start"], start_gap["t_end"]])
            injections.append(entry)
            continue

        if len(t_seg) < 2:
            entry["integrable"] = False
            add_finding("error", "empty_integration_window",
                        "注射后没有可用采样点", f"/injections/row/{pos}/time")
            injections.append(entry)
            continue

        ei, t_end, kind = _endpoint(t_seg, y_seg, band, int(pol["keep_samples"]), next_t)
        tw, yw = t_seg[: ei + 1], y_seg[: ei + 1]
        heat_ucal = float(np.trapezoid(yw, tw)) if hasattr(np, "trapezoid") else float(np.trapz(yw, tw))
        heat = heat_ucal * 1e-6  # µcal -> cal

        entry["window"]["t_end"] = float(tw[-1])
        entry["window"]["i_end"] = int(np.where(t == tw[-1])[0][0])
        entry["heat_cal"] = heat
        entry["heat_per_volume"] = heat / inj["volume"] if inj["volume"] else None
        entry["endpoint_kind"] = kind
        entry["window_overlaps_next"] = kind == "overlaps_next"

        if gap_block is not None:
            entry["clipped_by_gap"] = True
            entry["gap_ids"].append(gap_block["gap_id"])
            add_finding("warning", "window_clipped_by_gap",
                        f"积分窗被 {gap_block['gap_id']} 夹断，未跨越缺口",
                        f"/injections/row/{pos}/window",
                        gap_id=gap_block["gap_id"],
                        clipped_at=float(tw[-1]),
                        gap_interval=[gap_block["t_start"], gap_block["t_end"]])
        if kind == "overlaps_next":
            add_finding("info", "overlapping_peaks",
                        f"注射 {inj['index']} 的峰与下一注射重叠，窗口在下一注射时刻截止，积分仍然有效",
                        f"/injections/row/{pos}/window",
                        next_injection_time=next_t)
        elif kind == "end_of_trace":
            add_finding("warning", "endpoint_not_returned",
                        "曲线结束时仍未回到基线带，端点取曲线末点",
                        f"/injections/row/{pos}/window")
        injections.append(entry)

    if skipped:
        findings.append({
            "finding_id": "run::unparseable_samples",
            "severity": "warning", "code": "unparseable_samples",
            "message": f"{len(skipped)} 个采样点无法解析，已留在原始材料并按缺口/跳过处理",
            "loc": "/power", "row_indices": skipped,
        })

    return {
        "kind": "integration_run",
        "policy": pol,
        "inputs": {
            "power_samples": len(t),
            "unparseable_samples": skipped,
            "injection_count": len(injs),
            "baseline_mode": baseline_material.get("mode") if baseline_material else None,
        },
        "time_axis_unit": "s",
        "power_axis_unit": "µcal/s",
        "energy_unit": "cal",
        "noise_sigma": noise,
        "baseline_band": band,
        "median_sample_dt": median_dt,
        "gaps": gaps,
        "injections": injections,
        "findings": findings,
    }
