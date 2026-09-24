"""Per-injection integration of raw ITC power curves.

Domain rules implemented here:

* post-injection peaks may overlap each other; each injection still owns an
  independent integration window (flag ``overlapping_peaks``);
* an integration window never crosses a data gap: it is split into fragments
  at gaps and a window without an anchorable, sample-bearing fragment is
  recorded invalid with a position-bearing diagnostic (never a silent value);
* the first injection can be discarded from fitting without deleting its
  integral (``first_injection_action == "discard"``);
* endpoint detection supports baseline-return (auto), fixed delay and
  next-injection rules; endpoints are stored as explicit evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import Diagnostic, SemanticUnbound
from .units import convert_series


@dataclass(frozen=True)
class IntegrationParams:
    endpoint_rule: str = "auto"  # auto | fixed_delay | next_injection
    delay_s: float = 300.0
    auto_sigma: float = 3.0
    first_injection_action: str = "keep"  # keep | discard
    gap_threshold_s: float = 10.0  # missing-sample interval larger than this is a gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_rule": self.endpoint_rule,
            "delay_s": self.delay_s,
            "auto_sigma": self.auto_sigma,
            "first_injection_action": self.first_injection_action,
            "gap_threshold_s": self.gap_threshold_s,
        }


def _diag(code: str, message: str, **where: Any) -> dict[str, Any]:
    return Diagnostic(code, message, where).to_dict()


def normalized_series(power_material: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Convert the raw material to ordered, base-unit arrays (derived layer)."""
    content = power_material["content"]
    time_unit = content["time_unit"]
    value_unit = content["value_unit"]
    diagnostics: list[dict[str, Any]] = []

    if any(v is None for v in content["times"]) or any(v is None for v in content["values"]):
        diagnostics.append(
            _diag(
                "NON_NUMERIC_SAMPLES_RETAINED",
                "non-convertible samples stay in the raw material; excluded from derived series",
                role="power",
                scope=power_material.get("scope"),
            )
        )
    pairs = [
        (tv, pv)
        for tv, pv in zip(content["times"], content["values"])
        if tv is not None and pv is not None and np.isfinite(tv) and np.isfinite(pv)
    ]
    if len(pairs) < 2:
        raise SemanticUnbound(
            "power curve has fewer than two usable samples",
            [Diagnostic("TOO_FEW_SAMPLES", "need >=2 finite numeric samples", {"role": "power"})],
        )
    t_arr = np.array(convert_series([p[0] for p in pairs], time_unit, "time"), dtype=float)
    p_arr = np.array(convert_series([p[1] for p in pairs], value_unit, "power"), dtype=float)
    order = np.argsort(t_arr, kind="mergesort")
    return t_arr[order], p_arr[order], diagnostics


def detect_gaps(t: np.ndarray, content: dict[str, Any], threshold_s: float) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    median_dt = float(np.median(np.diff(t))) if len(t) >= 2 else float("inf")
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        if dt > max(threshold_s, 5.0 * median_dt):
            gaps.append({"start_s": float(t[i]), "end_s": float(t[i + 1]), "kind": "temporal"})
    for line in content.get("gap_marker_lines") or []:
        gaps.append({"at_line": line, "kind": "marker"})
    gaps.sort(key=lambda g: (g.get("start_s", -np.inf), g.get("at_line", 0)))
    return gaps


def _fit_baseline(t: np.ndarray, p: np.ndarray, t0: float, t1: float, pre_s: float, post_s: float, order: int):
    pre_mask = (t >= t0 - pre_s) & (t < t0)
    post_mask = (t > t1) & (t <= t1 + post_s)
    anchor_t = np.concatenate([t[pre_mask], t[post_mask]])
    anchor_p = np.concatenate([p[pre_mask], p[post_mask]])
    if anchor_t.size < max(2, order + 1):
        return None, anchor_t.size
    if order == 0:
        return (float(np.mean(anchor_p)),), anchor_t.size
    coeff = np.polyfit(anchor_t, anchor_p, 1)
    return (float(coeff[0]), float(coeff[1])), anchor_t.size


def _baseline_at(coeff, order: int, t: np.ndarray) -> np.ndarray:
    if order == 0:
        return np.full_like(t, coeff[0], dtype=float)
    return coeff[0] * t + coeff[1]


def _auto_endpoint(t: np.ndarray, p: np.ndarray, start: int, t0: float, sigma: float) -> float | None:
    """First time after ``t0`` where deviation stays within sigma for >=5 s."""
    pre = p[t < t0]
    level = float(np.std(pre)) if pre.size >= 2 else float(np.std(p))
    if not np.isfinite(level) or level == 0:
        level = max(1e-9, 0.02 * float(np.max(np.abs(p[t >= t0]))))
    near = np.abs(p) <= max(sigma * level, 1e-9)
    hold_s = 5.0
    n = len(t)
    idx = start
    while idx < n:
        if near[idx]:
            j = idx
            while j < n and near[j] and t[j] - t[idx] < hold_s:
                j += 1
            if t[j - 1] - t[idx] >= hold_s:
                return float(t[idx])
        idx += 1
    return None


def _split_fragments(seg_t: np.ndarray, seg_p: np.ndarray, crossing: list[tuple[float, float]]):
    if not crossing:
        return [(seg_t, seg_p)] if seg_t.size else []
    inside = np.zeros(seg_t.size, dtype=bool)
    for a, b in crossing:
        inside |= (seg_t > a) & (seg_t < b)
    fragments = []
    cur_t: list[float] = []
    cur_p: list[float] = []
    for tv, pv, is_gap in zip(seg_t, seg_p, inside):
        if is_gap:
            if cur_t:
                fragments.append((np.array(cur_t), np.array(cur_p)))
                cur_t, cur_p = [], []
        else:
            cur_t.append(float(tv))
            cur_p.append(float(pv))
    if cur_t:
        fragments.append((np.array(cur_t), np.array(cur_p)))
    return fragments


def integrate(
    power_material: dict[str, Any],
    baseline_material: dict[str, Any],
    injections_material: dict[str, Any],
    params: IntegrationParams,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    t, p, series_diags = normalized_series(power_material)
    diagnostics.extend(series_diags)

    inj = injections_material["content"]
    inj_times = np.array(convert_series([r["time"] for r in inj["rows"]], inj["time_unit"], "time"))
    inj_volumes = np.array(convert_series([r["volume"] for r in inj["rows"]], inj["volume_unit"], "volume"))

    base = baseline_material["content"]
    pre_s = float(base["anchor_pre_s"]["base"])
    post_s = float(base["anchor_post_s"]["base"])
    order = int(base["order"])

    gaps = detect_gaps(t, power_material["content"], params.gap_threshold_s)
    gintervals = [(g["start_s"], g["end_s"]) for g in gaps if g["kind"] == "temporal"]

    # Global residual (constant level removed) is used only for auto-endpoint.
    global_level = float(np.polyfit(t, p, 0)[0])
    residual = p - global_level

    results: list[dict[str, Any]] = []
    for k, row in enumerate(inj["rows"]):
        t0 = float(inj_times[k])
        next_t0 = float(inj_times[k + 1]) if k + 1 < len(inj_times) else None
        start_idx = int(np.searchsorted(t, t0, side="left"))

        if params.endpoint_rule == "fixed_delay":
            t1 = t0 + params.delay_s
        elif params.endpoint_rule == "next_injection":
            t1 = next_t0 if next_t0 is not None else float(t[-1])
        else:
            auto = _auto_endpoint(t, residual, start_idx, t0, params.auto_sigma)
            t1 = auto if auto is not None else float(t[-1])
        if next_t0 is not None:
            t1 = min(t1, next_t0)
        if t1 <= t0:
            t1 = t0

        crossing = [(a, b) for a, b in gintervals if b > t0 and a < t1]
        end_idx = int(np.searchsorted(t, t1, side="right"))
        window_t = t[start_idx:end_idx]
        window_p = p[start_idx:end_idx]
        mask = np.ones(window_t.size, dtype=bool)
        for a, b in crossing:
            mask &= ~((window_t > a) & (window_t < b))
        window_t, window_p = window_t[mask], window_p[mask]

        injection_diags: list[dict[str, Any]] = []
        area: float | None = None
        fragment_edges: list[tuple[float, float]] = []
        coeff, n_anchor = _fit_baseline(t, p, t0, t1, pre_s, post_s, order)

        if coeff is None:
            injection_diags.append(
                _diag(
                    "INSUFFICIENT_BASELINE_ANCHORS",
                    "not enough anchor samples around the integration window",
                    injection=row.get("index_raw", str(k + 1)),
                )
            )
        if window_t.size < 2:
            injection_diags.append(
                _diag(
                    "WINDOW_NO_SAMPLES",
                    "integration window contains fewer than two samples outside gaps",
                    injection=row.get("index_raw", str(k + 1)),
                )
            )
        else:
            fragments = _split_fragments(window_t, window_p, crossing)
            if coeff is not None:
                area = 0.0
                for ft, fp in fragments:
                    area += float(np.trapezoid(fp - _baseline_at(coeff, order, ft), ft))
                    fragment_edges.append((float(ft[0]), float(ft[-1])))
                if not fragment_edges:
                    area = None
                    injection_diags.append(
                        _diag(
                            "NO_VALID_WINDOW_FRAGMENT",
                            "window contains no samples outside gaps",
                            injection=row.get("index_raw", str(k + 1)),
                        )
                    )

        pre_level = p[(t >= t0 - pre_s) & (t < t0)]
        post_level = p[(t > t1) & (t <= t1 + post_s)]
        baseline_level = (
            float(np.mean(np.concatenate([pre_level, post_level])))
            if pre_level.size + post_level.size
            else None
        )
        peak_mask = (t >= t0) & (t <= t1)
        has_peak = (
            baseline_level is not None
            and bool(np.any(peak_mask))
            and bool(np.max(np.abs(p[peak_mask] - baseline_level)) > 5e-3)
        )
        overlapping = next_t0 is not None and t1 >= next_t0

        status = "valid" if area is not None else "invalid"
        results.append(
            {
                "injection_index": int(row["index"]) if row.get("index") is not None else k + 1,
                "injection_time_s": t0,
                "injection_volume_uL": float(inj_volumes[k]),
                "window_start_s": t0,
                "window_end_s": float(t1),
                "endpoint_rule": params.endpoint_rule,
                "fragments_s": fragment_edges,
                "crossed_gaps": [{"start_s": a, "end_s": b} for a, b in crossing],
                "n_baseline_anchors": int(n_anchor),
                "peak_present": has_peak,
                "overlapping_peaks": overlapping,
                "first_injection": k == 0,
                "discarded": params.first_injection_action == "discard" and k == 0,
                "baseline_order": order,
                "area_uJ": area,
                "status": status,
                "diagnostics": injection_diags,
            }
        )

    if params.first_injection_action == "discard":
        diagnostics.append(
            _diag(
                "FIRST_INJECTION_DISCARDED",
                "first injection retained for review but excluded from model fitting",
                injection=results[0]["injection_index"],
            )
        )
    for r in results:
        if r["status"] == "invalid":
            diagnostics.extend(r["diagnostics"])

    return {
        "results": results,
        "params": params.to_dict(),
        "gaps": gaps,
        "diagnostics": diagnostics,
        "summary": {
            "n_injections": len(results),
            "n_valid": sum(r["status"] == "valid" for r in results),
            "n_discarded": sum(r["discarded"] for r in results),
            "n_overlapping": sum(r["overlapping_peaks"] for r in results),
            "n_gap_crossed": sum(1 for r in results if r["crossed_gaps"]),
        },
    }
