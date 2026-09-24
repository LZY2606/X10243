"""Per-injection integration engine.

Hard rules exercised by the acceptance suite:
* post-injection peaks may overlap; a window may terminate at the next
  injection start, but it never double-takes another window's interval;
* an integration window never crosses a data gap - it terminates at the last
  valid point, or is reported impossible when no valid span exists;
* dilution-heat subtraction carries an explicit correspondence rule;
* the first injection can be discarded for fitting while keeping its raw
  integral.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import ResourceExhausted, SemanticImpossible
from .materials import MaterialVersion
from .units import from_canonical

W_DEFAULTS = {"pre": 2.0, "post": 120.0, "max_points": 200000}


def valid_spans(t: np.ndarray, *, max_gap_s: float) -> list[tuple[int, int]]:
    """Return contiguous index spans whose time gaps stay within max_gap_s.

    A span is [i0, i1) over valid samples only.
    """
    if t.size == 0:
        return []
    spans = []
    start = 0
    diffs = np.diff(t)
    for i, gap in enumerate(diffs, start=1):
        if gap > max_gap_s:
            spans.append((start, i))
            start = i
    spans.append((start, t.size))
    return spans


def _span_for(spans: list[tuple[int, int]], t: np.ndarray, t0: float, t1: float
              ) -> tuple[int, int] | None:
    for i0, i1 in spans:
        if t[i0] <= t0 and t[i1 - 1] >= t0:
            return i0, i1
        # window may start slightly before first sample of a span
        if t[i0] <= t1 and t[i1 - 1] >= t0 and t[i0] >= t0:
            return i0, i1
    return None


def integrate_shot(
    shot_index: int,
    t0: float,
    next_injection_s: float | None,
    power: MaterialVersion,
    baseline: MaterialVersion | None,
    *,
    pre_s: float = W_DEFAULTS["pre"],
    post_s: float = W_DEFAULTS["post"],
    max_gap_s: float = 5.0,
    endpoint_override: tuple[float, float] | None = None,
    max_points: int = W_DEFAULTS["max_points"],
    start_at_injection: bool = True,
) -> dict:
    tp, vp = power.series.arrays()
    if tp.size < 2:
        raise SemanticImpossible(
            "power curve has fewer than two valid points",
            diagnostics=[{"code": "insufficient_points", "shot": shot_index}],
        )
    tb, vb = (baseline.series.arrays() if baseline else (None, None))
    win_start, win_end = (t0 - pre_s, t0 + post_s)
    if start_at_injection:
        win_start = t0
    if endpoint_override:
        win_start, win_end = endpoint_override
    overlap_with = None
    if next_injection_s is not None and win_end > next_injection_s:
        overlap_with = next_injection_s
        win_end = next_injection_s
    if win_end <= win_start:
        raise SemanticImpossible(
            f"shot {shot_index}: integration window is empty after overlap clipping",
            diagnostics=[{"code": "empty_window", "shot": shot_index,
                          "window_start_s": win_start, "window_end_s": win_end}],
        )
    spans = valid_spans(tp, max_gap_s=max_gap_s)
    span = _span_for(spans, tp, t0, win_end)
    diagnostics: list[dict] = []
    gap_terminated = False
    if span is None:
        raise SemanticImpossible(
            f"shot {shot_index}: no valid data span contains injection time {t0}s",
            diagnostics=[{"code": "window_without_span", "shot": shot_index,
                          "injection_time_s": t0}],
        )
    i0, i1 = span
    st = tp[i0:i1]
    sv = vp[i0:i1]
    sel = (st >= win_start) & (st <= win_end)
    wt, wv = st[sel], sv[sel]
    if wt.size < 2:
        raise SemanticImpossible(
            f"shot {shot_index}: fewer than two valid samples inside window",
            diagnostics=[{"code": "window_without_span", "shot": shot_index,
                          "window_start_s": win_start, "window_end_s": win_end}],
        )
    if endpoint_override is None and wt[-1] < win_end - 1e-9:
        gap_terminated = True
        diagnostics.append({"code": "window_terminated_at_gap", "shot": shot_index,
                            "requested_end_s": win_end, "effective_end_s": float(wt[-1])})
    if wt.size > max_points:
        raise ResourceExhausted(
            f"shot {shot_index}: integration point budget exceeded",
            budget="integration_points", used=wt.size, limit=max_points,
        )
    if tb is not None and tb.size:
        bv = np.interp(wt, tb, vb)
    else:
        bv = np.zeros_like(wv)
    corrected = wv - bv
    area_j = float(np.trapezoid(corrected, wt)) if hasattr(np, "trapezoid") \
        else float(np.trapz(corrected, wt))
    effective_end = float(wt[-1])
    return {
        "index": shot_index,
        "injection_time_s": float(t0),
        "window": {"start_s": float(wt[0]), "end_s": effective_end,
                   "requested_start_s": float(win_start),
                   "requested_end_s": float(win_end),
                   "n_points": int(wt.size),
                   "time_basis": "experiment_seconds",
                   "power_unit": "W", "integral_unit": "J"},
        "integral_j": area_j,
        "baseline_kind": "curve" if baseline is not None else "none",
        "overlap_next_injection_s": overlap_with,
        "gap_terminated": gap_terminated,
        "diagnostics": diagnostics,
    }


def blank_integrals(blank: MaterialVersion | None, blank_schedule, *,
                    pre_s: float, post_s: float, baseline=None,
                    max_gap_s: float = 5.0) -> dict[int, dict]:
    """Integrate blank injections on the same baseline basis as the sample.

    Passing the sample baseline curve keeps sample and dilution heats on one
    reference (power - baseline); otherwise absolute basins would differ.
    """
    if blank is None or not blank_schedule:
        return {}
    out: dict[int, dict] = {}
    injections = sorted(blank_schedule, key=lambda i: i.time_s)
    times = [i.time_s for i in injections]
    for pos, inj in enumerate(injections):
        nxt = times[pos + 1] if pos + 1 < len(times) else None
        try:
            out[inj.index] = integrate_shot(
                inj.index, inj.time_s, nxt, blank, baseline,
                pre_s=pre_s, post_s=post_s, max_gap_s=max_gap_s)
        except SemanticImpossible as exc:
            out[inj.index] = {"index": inj.index, "diagnostics": exc.diagnostics,
                              "missing": True}
    return out


def apply_blank_rule(shots: list[dict], blanks: dict[int, dict], rule: str,
                     *, volume_l: float, blank_volume_l: float | None) -> list[dict]:
    """Attach dilution heats using a declared correspondence rule.

    rules:
      * by_index   - blank shot k corresponds to sample shot k
      * volume_proportional - scaled by injected volume ratio
      * none       - no dilution subtraction
    Missing correspondence is never guessed: the shot keeps its raw integral
    and receives an explicit blank_missing diagnostic.
    """
    out = []
    for shot in shots:
        s = dict(shot)
        s["blank_rule"] = rule
        s["dilution_j"] = None
        s["net_j"] = s["integral_j"]
        if rule == "none":
            out.append(s)
            continue
        b = blanks.get(s["index"])
        if not b or b.get("missing"):
            s["diagnostics"] = s.get("diagnostics", []) + [{
                "code": "blank_correspondence_missing", "shot": s["index"],
                "rule": rule,
                "message": f"no blank corresponding to shot {s['index']} under {rule}",
            }]
            s["blank_status"] = "missing"
            out.append(s)
            continue
        q = b["integral_j"]
        if rule == "volume_proportional":
            denom = blank_volume_l if blank_volume_l else volume_l
            q = q * (volume_l / denom)
            if blank_volume_l is None:
                s["diagnostics"] = s.get("diagnostics", []) + [{
                    "code": "blank_volume_assumed_equal", "shot": s["index"],
                    "rule": rule}]
        elif rule != "by_index":
            raise SemanticImpossible(f"unknown blank correspondence rule {rule!r}")
        s["dilution_j"] = float(q)
        s["net_j"] = float(s["integral_j"] - q)
        s["blank_status"] = "applied"
        out.append(s)
    return out


def heat_per_mole(shots: list[dict], schedule, *, use_net: bool = True) -> None:
    vols = {i.index: i.volume_l for i in schedule}
    for s in shots:
        vol = vols[s["index"]]
        q = s["net_j"] if use_net else s["integral_j"]
        s["injection_volume_l"] = vol
        s["heat_per_injection_j"] = q
