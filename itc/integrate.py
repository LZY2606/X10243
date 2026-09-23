"""Injection integration over raw power samples.

Domain rules implemented here:
  - post-injection peaks MAY overlap: a window is not clipped by the next
    injection time, and overlaps are recorded explicitly
  - an integration window NEVER crosses a data gap; it is truncated at the
    gap edge and a positioned diagnostic is attached
  - window endpoints are interpolated onto the sampling grid and reported
    with their coordinate basis (seconds, uJ/s)
  - nothing is silently dropped: injections that cannot be integrated get a
    status + diagnostic instead of a fabricated number
"""
from __future__ import annotations

import numpy as np

from .diagnostics import Diagnostic

PARAMS_VERSION = "integrate/1"
DEFAULT_PARAMS = {
    "pre_s": 2.0,            # window starts this many seconds before the injection
    "max_window_s": 180.0,   # hard cap on window length
    "settle_tol_frac": 0.03, # settled when |P-baseline| < frac * peak amplitude ...
    "settle_hold_s": 15.0,   # ... sustained for this long
    "gap_factor": 3.0,       # sampling interval > factor * median(dt) is a data gap
}


def detect_gaps(times: np.ndarray, factor: float = DEFAULT_PARAMS["gap_factor"]) -> list:
    """Return list of {start_s, end_s} gaps in the sampling grid."""
    if len(times) < 3:
        return []
    dts = np.diff(times)
    med = float(np.median(dts))
    gaps = []
    for i, dt in enumerate(dts):
        if dt > factor * med:
            gaps.append({"start_s": float(times[i]), "end_s": float(times[i + 1])})
    return gaps


def _interp(t: np.ndarray, y: np.ndarray, x: float) -> float:
    return float(np.interp(x, t, y))


def _window_integral(t: np.ndarray, y: np.ndarray, start: float, end: float) -> float:
    """Trapezoid integral over [start, end] with linearly interpolated endpoints."""
    mask = (t > start) & (t < end)
    ts = np.concatenate(([start], t[mask], [end]))
    ys = np.concatenate(([_interp(t, y, start)], y[mask], [_interp(t, y, end)]))
    return float(np.trapezoid(ys, ts))


def _find_end(times, resid, t0, peak_amp, params, hard_end):
    """First time after t0 where |resid| stays below tol for settle_hold_s."""
    tol = params["settle_tol_frac"] * max(abs(peak_amp), 1e-9)
    hold = params["settle_hold_s"]
    idx = np.searchsorted(times, t0)
    below_since = None
    for j in range(idx, len(times)):
        if abs(resid[j]) <= tol:
            if below_since is None:
                below_since = times[j]
            if times[j] - below_since >= hold:
                return min(below_since + hold, hard_end)
        else:
            below_since = None
    return hard_end


def integrate_run(times, powers, baseline, injections, params=None) -> dict:
    """Integrate every injection. Returns per-injection results + diagnostics.

    times/powers/baseline: canonical arrays (s, uJ/s, uJ/s).
    injections: list of {"index", "time_s", "volume_uL"} (already canonical).
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    times = np.asarray(times, dtype=float)
    powers = np.asarray(powers, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    resid = powers - baseline
    gaps = detect_gaps(times, p["gap_factor"])
    results = []
    for inj in injections:
        idx = inj["index"]
        t0 = float(inj["time_s"])
        uid = f"inj:{idx:02d}"
        diags: list[Diagnostic] = []
        path = f"injections[{idx}]"
        inside_gap = next((g for g in gaps if g["start_s"] < t0 < g["end_s"]), None)
        if inside_gap:
            diags.append(Diagnostic(
                code="injection_in_data_gap",
                message=f"injection at t={t0}s falls inside data gap "
                        f"[{inside_gap['start_s']}, {inside_gap['end_s']}]s; not integrated",
                path=path))
            results.append({"uid": uid, "index": idx, "time_s": t0,
                            "volume_uL": inj["volume_uL"], "status": "invalid",
                            "heat_uJ": None, "window": None,
                            "diagnostics": [d.to_dict() for d in diags]})
            continue
        start = max(t0 - p["pre_s"], float(times[0]))
        peak_amp = float(np.max(np.abs(resid[times >= t0]))) if np.any(times >= t0) else 0.0
        hard_end = min(t0 + p["max_window_s"], float(times[-1]))
        end = _find_end(times, resid, t0, peak_amp, p, hard_end)
        overlaps_next = False
        if idx + 1 < len(injections):
            nxt = float(injections[idx + 1]["time_s"])
            if end > nxt:
                overlaps_next = True  # allowed: peaks may overlap; recorded, not clipped
        truncated_by_gap = None
        for g in gaps:
            if start < g["start_s"] < end:
                end = g["start_s"]  # window must not cross a data gap
                truncated_by_gap = g
        if truncated_by_gap:
            diags.append(Diagnostic(
                code="window_truncated_by_gap",
                message=f"integration window truncated at data gap edge "
                        f"{truncated_by_gap['start_s']}s (gap extends to {truncated_by_gap['end_s']}s)",
                path=f"{path}.window", severity="warning"))
        if overlaps_next:
            diags.append(Diagnostic(
                code="window_overlaps_next_injection",
                message="integration window overlaps the next injection peak (allowed)",
                path=f"{path}.window", severity="info"))
        heat = _window_integral(times, resid, start, end)
        results.append({
            "uid": uid, "index": idx, "time_s": t0, "volume_uL": inj["volume_uL"],
            "status": "ok" if not truncated_by_gap else "truncated",
            "heat_uJ": heat,
            "window": {
                "start": {"t_s": start, "power_uJs": _interp(times, powers, start),
                          "basis": "elapsed_s", "endpoint": "interpolated"},
                "end": {"t_s": end, "power_uJs": _interp(times, powers, end),
                        "basis": "elapsed_s", "endpoint": "interpolated"},
                "overlaps_next": overlaps_next,
                "truncated_by_gap": bool(truncated_by_gap),
            },
            "diagnostics": [d.to_dict() for d in diags],
        })
    return {"params": p, "params_version": PARAMS_VERSION,
            "gaps": gaps, "injections": results}
