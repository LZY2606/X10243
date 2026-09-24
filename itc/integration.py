"""Injection-by-injection integration of ITC power curves.

Hard rules (the "really tricky" domain rules):

* Post-injection peaks may overlap; each peak is integrated independently and
  flagged ``overlaps_next`` / ``overlaps_prev``.
* An integration window never crosses a data gap: explicit gaps, null samples
  and series end are all hard endpoint boundaries. A window that cannot close
  returns ``gap_inside`` / ``incomplete`` diagnostics instead of fabricated
  points.
* Baseline subtraction uses the separately-versioned baseline material.
* Dilution-heat (blank) subtraction requires an explicit correspondence rule
  plus a matching blank injection; otherwise only the raw heat is reported
  with ``blank_match_missing``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ident import make_id
from .units import require

DEFAULT_PRE = 5.0          # seconds of pre-peak baseline window
DEFAULT_RETURN_TOL = 5.0   # uW tolerance to baseline for window closing
RETURN_DWELL = 3           # consecutive near-baseline samples to "close"


@dataclass
class Series:
    t: np.ndarray
    p: np.ndarray            # NaN marks missing samples
    gaps: list[tuple[float, float]] = field(default_factory=list)


def _qval(node: dict | None):
    if not node or node.get("value") is None:
        return None
    if any(d.get("severity") == "error" for d in node.get("diagnostics", [])):
        return None
    return float(node["value"])


def build_series(power_content: dict[str, Any]) -> Series:
    samples = power_content.get("samples", [])
    t = np.array([s["t"] for s in samples], dtype=float)
    p = np.array([np.nan if s.get("p") is None else s["p"] for s in samples],
                 dtype=float)
    gaps = [(float(g["start"]), float(g["end"]))
            for g in power_content.get("gaps", [])
            if g.get("start") is not None and g.get("end") is not None]
    return Series(t=t, p=p, gaps=gaps)


def baseline_curve(baseline_content: dict[str, Any] | None,
                   t: np.ndarray) -> np.ndarray | None:
    if not baseline_content:
        return None
    kind = baseline_content.get("kind", "constant")
    if kind == "constant":
        val = _qval(baseline_content.get("value"))
        return None if val is None else np.full_like(t, val)
    if kind == "linear":
        a = _qval(baseline_content.get("intercept"))
        b = _qval(baseline_content.get("slope"))
        if a is None or b is None:
            return None
        return a + b * t
    if kind == "curve" and baseline_content.get("samples"):
        pts = baseline_content["samples"]
        bt = np.array([s["t"] for s in pts], dtype=float)
        bp = np.array([np.nan if s.get("p") is None else s["p"]
                       for s in pts], dtype=float)
        good = ~np.isnan(bp)
        if good.sum() < 2:
            return None
        return np.interp(t, bt[good], bp[good])
    return None


def _gap_for(series: Series, lo: float, hi: float) -> tuple[float, float] | None:
    for a, b in series.gaps:
        if a < hi and b > lo:
            return (max(a, lo), min(b, hi))
    mask = (series.t > lo) & (series.t < hi)
    if mask.any() and np.isnan(series.p[mask]).any():
        idx = int(np.argmax(mask & np.isnan(series.p)))
        return (float(series.t[idx]), float(series.t[idx]))
    return None


def _next_boundary(series: Series, t0: float) -> float:
    """Earliest hard endpoint after t0: next gap start or last sample time."""
    end = float(series.t[-1]) if len(series.t) else t0
    for a, _b in series.gaps:
        if a > t0:
            end = min(end, a)
    nans = np.where(np.isnan(series.p) & (series.t > t0))[0]
    if len(nans):
        end = min(end, float(series.t[nans[0]]))
    return end


def integrate_injection(series: Series, injection: dict, next_time: float | None,
                        baseline: np.ndarray | None,
                        *, pre: float = DEFAULT_PRE,
                        tol: float = DEFAULT_RETURN_TOL) -> dict[str, Any]:
    """Integrate a single injection peak.

    Returns a structured result (never raises for expected domain issues).
    Energy unit: uW*s = uJ.
    """
    t0 = _qval(injection.get("time"))
    volume = _qval(injection.get("volume"))
    index = injection.get("index")
    diagnostics: list[dict[str, str]] = []
    status = "ok"
    if t0 is None:
        return {"index": index, "status": "invalid",
                "diagnostics": [{"code": "missing_time",
                                 "message": "injection time unavailable",
                                 "location": f"/injections/{index}/time",
                                 "severity": "error"}]}
    if volume is None:
        diagnostics.append({"code": "missing_volume",
                            "message": "injection volume unavailable",
                            "location": f"/injections/{index}/volume",
                            "severity": "error"})
        status = "invalid"

    hard_end = _next_boundary(series, t0)
    natural_end = next_time if next_time is not None else hard_end
    window_end_limit = min(hard_end, natural_end)
    overlaps_next = False
    if next_time is not None and next_time < hard_end:
        overlaps_next = True

    after = (series.t >= t0) & (series.t < window_end_limit)
    ta, pa = series.t[after], series.p[after]
    base_a = baseline[after] if baseline is not None else np.zeros_like(ta)
    pre_mask = (series.t >= t0 - pre) & (series.t < t0)
    pre_base = float(np.nanmean(series.p[pre_mask])) if pre_mask.any() \
        and not np.all(np.isnan(series.p[pre_mask])) else (
        float(base_a[0]) if len(base_a) else 0.0)

    end_idx = None
    if len(ta):
        near = np.abs(pa - (base_a + pre_base if baseline is None else base_a)) <= tol
        dwell = 0
        for i, flag in enumerate(near):
            dwell = dwell + 1 if flag else 0
            if dwell >= RETURN_DWELL:
                end_idx = i
                break
    if end_idx is None:
        if overlaps_next:
            end_idx = len(ta) - 1 if len(ta) else None
            diagnostics.append({"code": "peak_truncated",
                                "message": "peak still open at next injection; "
                                           "window truncated (peaks overlap)",
                                "location": f"/power/injections/{index}",
                                "severity": "warning"})
        elif hard_end <= (natural_end if natural_end != hard_end else hard_end) and \
                hard_end < float(series.t[-1] if len(series.t) else hard_end):
            end_idx = len(ta) - 1 if len(ta) else None
            status = "gap_inside"
            diagnostics.append({"code": "gap_inside",
                                "message": "integration window cannot close "
                                           "before a data gap",
                                "location": f"/power/gaps@{hard_end:.3g}s",
                                "severity": "error"})
        elif (not len(ta)) or window_end_limit >= float(
                series.t[-1]) and end_idx is None:
            end_idx = len(ta) - 1 if len(ta) else None
            if status != "invalid":
                status = "incomplete"
            diagnostics.append({"code": "incomplete_window",
                                "message": "series ends before peak returns "
                                           "to baseline",
                                "location": f"/power/injections/{index}",
                                "severity": "error"})
        else:
            end_idx = len(ta) - 1
    q_raw = q_base = None
    window = None
    if len(ta) and end_idx is not None:
        tw, pw = ta[: end_idx + 1], pa[: end_idx + 1]
        bw = base_a[: end_idx + 1]
        if baseline is None:
            bw = np.full_like(tw, pre_base)
        q_raw = float(np.trapezoid(pw - np.full_like(tw, pre_base)
                                   if baseline is None else pw - bw, tw))
        q_base = float(np.trapezoid(pw - bw, tw))
        window = {"start": float(t0), "end": float(tw[-1]),
                  "hard_end": float(hard_end),
                  "start_time_basis": "injection_time",
                  "crosses_gap": False}
    gap = _gap_for(series, t0,
                   window["end"] if window else window_end_limit)
    if gap is not None and status == "ok":
        status = "gap_inside"
        diagnostics.append({"code": "gap_inside",
                            "message": "integration window contains a data gap",
                            "location": f"/power/gaps/{gap[0]:.3g}",
                            "severity": "error"})
        if window:
            window["crosses_gap"] = True
    if overlaps_next and status == "ok":
        status = "overlapping"
    result = {
        "index": index, "status": status,
        "time_s": t0, "volume_uL": volume,
        "window": window,
        "q_raw_uJ": q_raw, "q_baseline_uJ": q_base,
        "overlaps_next": overlaps_next,
        "diagnostics": diagnostics,
        "units": {"time": "s", "power": "uW", "volume": "uL", "energy": "uJ"},
    }
    return result


def integrate_all(power_content: dict[str, Any],
                  injections_content: dict[str, Any],
                  baseline_content: dict[str, Any] | None) -> list[dict[str, Any]]:
    series = build_series(power_content)
    baseline = baseline_curve(baseline_content, series.t)
    rows = injections_content.get("rows", [])
    times = [_qval(r.get("time")) for r in rows]
    results = []
    for i, row in enumerate(rows):
        nxt = times[i + 1] if i + 1 < len(times) else None
        res = integrate_injection(series, row, nxt, baseline)
        if i > 0 and results[-1].get("overlaps_next"):
            res["overlaps_prev"] = True
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Blank / dilution heat correspondence
# ---------------------------------------------------------------------------

def _blank_rows(blank_content: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not blank_content or not blank_content.get("present"):
        return []
    inj = blank_content.get("injections") or {}
    return inj.get("rows", [])


def match_blank(index: int, sample_volume_uL: float | None,
                blank_rows: list[dict[str, Any]], rule: str
                ) -> tuple[dict | None, str | None]:
    """Return ``(blank_row, reason)``. reason is None on a clean match."""
    if not rule:
        return None, "no_rule"
    if not blank_rows:
        return None, "blank_absent"
    if rule == "by_index":
        row = next((r for r in blank_rows if r.get("index") == index), None)
        return (row, None) if row else (None, "no_match")
    if rule == "by_volume_nearest":
        if sample_volume_uL is None:
            return None, "sample_volume_missing"
        best = None
        best_d = None
        for r in blank_rows:
            v = _qval(r.get("volume"))
            if v is None:
                continue
            d = abs(v - sample_volume_uL)
            if best_d is None or d < best_d:
                best, best_d = r, d
        return (best, None) if best is not None else (None, "no_match")
    return None, "unknown_rule"


def subtract_blanks(results: list[dict[str, Any]],
                    blank_content: dict[str, Any] | None,
                    blank_power: dict[str, Any] | None,
                    blank_baseline: dict[str, Any] | None,
                    rule: str | None) -> list[dict[str, Any]]:
    """Annotate each sample integration with its dilution-heat subtraction.

    The rule and the matched blank injection are recorded on every result.
    A missing correspondence leaves ``q_net_uJ`` None with an explicit
    diagnostic - subtraction is never guessed.
    """
    rows = _blank_rows(blank_content)
    blank_results: dict[int, dict] = {}
    if blank_power is not None and rows:
        for br in integrate_all(blank_power, blank_content["injections"],
                                blank_baseline):
            if br.get("index") is not None:
                blank_results[br["index"]] = br
    effective_rule = rule or (blank_content or {}).get("match_rule") or ""
    for res in results:
        diag = res["diagnostics"]
        res["blank_rule"] = effective_rule or None
        res["blank_match"] = None
        res["q_blank_uJ"] = None
        res["q_net_uJ"] = None
        if res.get("q_baseline_uJ") is None:
            continue
        if not blank_content or not blank_content.get("present"):
            diag.append({"code": "blank_match_missing",
                         "message": "blank control not imported; dilution heat "
                                    "cannot be subtracted",
                         "location": f"/blank", "severity": "warning"})
            continue
        if not effective_rule:
            diag.append({"code": "blank_match_missing",
                         "message": "no blank correspondence rule declared",
                         "location": "/blank/match_rule", "severity": "warning"})
            continue
        brow, reason = match_blank(res["index"], res.get("volume_uL"), rows,
                                   effective_rule)
        if reason:
            diag.append({"code": "blank_match_missing",
                         "message": f"blank correspondence failed: {reason} "
                                    f"(rule={effective_rule})",
                         "location": f"/blank/injections/{res['index']}",
                         "severity": "warning"})
            continue
        bidx = brow["index"]
        bres = blank_results.get(bidx)
        res["blank_match"] = {"rule": effective_rule, "blank_index": bidx,
                              "blank_volume_uL": _qval(brow.get("volume"))}
        if bres is None or bres.get("q_baseline_uJ") is None:
            diag.append({"code": "blank_match_missing",
                         "message": f"blank injection {bidx} has no usable "
                                    "integral",
                         "location": f"/blank/power/{bidx}",
                         "severity": "warning"})
            continue
        res["q_blank_uJ"] = bres["q_baseline_uJ"]
        res["q_net_uJ"] = res["q_baseline_uJ"] - bres["q_baseline_uJ"]
    return results


def integration_input_digest(*refs: Any) -> str:
    from .ident import digest
    return digest(list(refs), length=20)
