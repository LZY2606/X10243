"""注射积分内核：重叠峰允许，积分窗绝不跨越数据缺口；端点检验。

输入全部为规范化材料，纯计算、无浏览器依赖，可被测试直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import numpy as np

from .errors import Diagnostic
from .identity import point_id, result_id
from .materials import BLANK, BASELINE, POWER, SCHEDULE
from .rules import BlankRules, IntegrationRules


def _diag(code: str, message: str, location: str) -> dict[str, Any]:
    return Diagnostic(code, message, location).to_dict()


@dataclass
class IntegrationOutcome:
    id: str
    injections: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    input_summary: dict[str, Any]
    parent_ids: list[str]
    rule_version: str
    rule_id: str
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "integration",
            "accepted": self.accepted,
            "rule_version": self.rule_version,
            "rule_id": self.rule_id,
            "input_summary": self.input_summary,
            "parent_ids": self.parent_ids,
            "injections": self.injections,
            "diagnostics": self.diagnostics,
        }


def _series(power: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    samples = power["samples"]
    t = np.array([s[0] for s in samples], dtype=float)
    p = np.array([np.nan if s[1] is None else s[1] for s in samples], dtype=float)
    mask = np.isfinite(p)
    t, p = t[mask], p[mask]
    order = np.argsort(t, kind="stable")
    return t[order], p[order]


def _baseline_at(baseline: dict[str, Any], t_query: np.ndarray) -> np.ndarray:
    anchors = sorted(baseline["anchors"], key=lambda a: a["t"])
    ta = np.array([a["t"] for a in anchors], dtype=float)
    good = [(a["t"], a["p"]) for a in anchors if a["p"] is not None]
    ta = np.array([g[0] for g in good], dtype=float)
    pa = np.array([g[1] for g in good], dtype=float)
    if len(ta) == 1:
        return np.full_like(t_query, pa[0])
    return np.interp(t_query, ta, pa)


def _gap_list(t: np.ndarray, threshold: float) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    if len(t) < 2:
        return out
    for i, d in enumerate(np.diff(t)):
        if float(d) > threshold:
            out.append({"after_t": float(t[i]), "before_t": float(t[i + 1]), "dt": float(d)})
    return out


def _integrate_one(
    t: np.ndarray,
    p: np.ndarray,
    base: np.ndarray,
    t0: float,
    t1_nominal: float,
    next_inj: float | None,
    gaps: list[dict[str, float]],
    rules: IntegrationRules,
    inj_index: int,
    global_noise: float,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    window_end = t1_nominal
    overlap = False
    if next_inj is not None and next_inj < t1_nominal:
        overlap = True
        diagnostics.append(_diag(
            "overlapping_peaks",
            f"第{inj_index}注峰与下一注射(t={next_inj:.1f}s)重叠；注射后峰允许重叠，窗按 {rules.overlap_policy} 处理",
            f"injections[{inj_index}].window"))
        if rules.overlap_policy == "partition":
            window_end = next_inj

    truncated_by_gap = False
    gap_inside = None
    for g in gaps:
        if t0 < g["after_t"] < window_end:
            window_end = min(window_end, g["after_t"])
            truncated_by_gap = True
            gap_inside = g
            diagnostics.append(_diag(
                "window_crosses_gap",
                f"积分窗在 {g['after_t']:.1f}s 遇到 {g['dt']:.1f}s 缺口；积分窗不跨越数据缺口，已在缺口前截断",
                f"injections[{inj_index}].window"))
            break

    mask = (t >= t0) & (t <= window_end)
    tw, pw, bw = t[mask], p[mask], base[mask]
    if len(tw) < 2:
        diagnostics.append(_diag("insufficient_points", "有效区间内样本不足，无法积分",
                                 f"injections[{inj_index}].window"))
        return {
            "index": inj_index,
            "t0": float(t0),
            "t1_nominal": float(t1_nominal),
            "t1_effective": float(window_end),
            "overlap": overlap,
            "truncated_by_gap": truncated_by_gap,
            "gap": gap_inside,
            "volume_l": None,
            "q": None,
            "q_net": None,
            "n_points": int(len(tw)),
            "endpoint_fail": None,
            "discarded": rules.discard_first and inj_index == 1,
            "blank_rule": None,
            "diagnostics": diagnostics,
            "point_ids": [],
        }

    corrected = pw - bw
    q = float(np.trapezoid(corrected, tw))

    tail_n = max(2, int(round(len(tw) * rules.tail_fraction)))
    tail = corrected[-tail_n:]
    noise = max(global_noise, 1e-12)
    # 端点检验：窗末段均值相对“零”（已减基线）的漂移，而非瞬时最大值
    tail_shift = float(np.mean(tail))
    tail_sem = noise / math.sqrt(len(tail))
    endpoint_fail = abs(tail_shift) > rules.tail_sigma * tail_sem
    if endpoint_fail:
        diagnostics.append(_diag(
            "endpoint_not_settled",
            f"窗末段均值 {tail_shift:.3e} cal/s 偏离基线 > {rules.tail_sigma}×SEM={rules.tail_sigma * tail_sem:.3e}，峰未回归基线",
            f"injections[{inj_index}].endpoint"))

    return {
        "index": inj_index,
        "t0": float(t0),
        "t1_nominal": float(t1_nominal),
        "t1_effective": float(window_end),
        "overlap": overlap,
        "truncated_by_gap": truncated_by_gap,
        "gap": gap_inside,
        "q": q,
        "q_net": None,
        "n_points": int(len(tw)),
        "tail_shift": tail_shift,
        "noise_sigma": noise,
        "endpoint_fail": endpoint_fail,
        "discarded": bool(rules.discard_first and inj_index == 1),
        "blank_rule": None,
        "diagnostics": diagnostics,
        "point_ids": [],
    }


def integrate_run(
    materials: dict,
    rules: IntegrationRules,
    *,
    is_blank: bool = False,
) -> IntegrationOutcome:
    if is_blank:
        block = materials[BLANK].canonical
        power = block["power_series"]
        schedule = block["injection_schedule"]
        baseline = block["baseline"]
    else:
        power = materials[POWER].canonical
        schedule = materials[SCHEDULE].canonical
        baseline = materials[BASELINE].canonical
    t, p = _series(power)
    base = _baseline_at(baseline, t)
    gaps = _gap_list(t, rules.gap_threshold_seconds)

    injections_src = sorted(schedule["injections"], key=lambda x: x["t"])
    times = [inj["t"] for inj in injections_src]
    diagnostics_all: list[dict[str, Any]] = []
    for g in gaps:
        diagnostics_all.append(_diag("data_gap",
            f"{g['dt']:.1f}s 数据缺口 ({g['after_t']:.1f}–{g['before_t']:.1f}s)",
            "power_series.samples"))

    results: list[dict[str, Any]] = []
    head_mask = (t >= 0.0) & (t < min(times[0], rules.baseline_fit_seconds)) if times else np.zeros_like(t, bool)
    head_resid = (p - base)[head_mask]
    global_noise = float(np.std(head_resid)) if len(head_resid) >= 2 else 1e-8
    for i, inj in enumerate(injections_src):
        t0 = inj["t"]
        next_inj = times[i + 1] if i + 1 < len(times) else None
        window = _integrate_one(t, p, base, t0, t0 + rules.window_seconds,
                                next_inj, gaps, rules, inj.get("index", i + 1), global_noise)
        window["volume_l"] = inj["volume"]
        results.append(window)

    if is_blank:
        parent_ids = sorted([materials[BLANK].id])
    else:
        parent_ids = sorted([materials[POWER].id, materials[SCHEDULE].id, materials[BASELINE].id])
    input_summary = {
        "power_samples": len(power["samples"]),
        "injections": len(injections_src),
        "t_span": [float(t[0]) if len(t) else None, float(t[-1]) if len(t) else None],
        "time_basis": "s",
        "heat_basis": "cal",
    }
    rid = result_id("integration", input_summary, rules.content(), parent_ids)
    for r in results:
        r["point_ids"] = [point_id(rid, f"injection-{r['index']}")]

    return IntegrationOutcome(
        id=rid,
        injections=results,
        diagnostics=diagnostics_all,
        input_summary=input_summary,
        parent_ids=parent_ids,
        rule_version=rules.version,
        rule_id=rules.id,
    )
