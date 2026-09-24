"""Injection integration engine.

Rules encoded here (mirrored as machine-readable diagnostics, not just logs):
  * post-injection peaks MAY overlap the next injection's rise; the window is
    clamped to the next injection start and the overlap is recorded;
  * an integration window must NOT cross a data gap: it is truncated at the
    gap edge, and an injection landing inside a gap is invalid;
  * the first injection can be discarded by configuration but stays on record.
"""
from __future__ import annotations

import numpy as np

from .domain import RESULT_BUDGET, RESULT_OK, diag

PARAM_VERSION = "integrate/1"
DEFAULT_CONFIG = {"pre_s": 2.0, "post_s": 45.0, "gap_factor": 5.0, "discard_first": True}

_trapz = getattr(np, "trapezoid", None) or np.trapz


def find_gaps(times, factor):
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return []
    dt = np.diff(times)
    med = float(np.median(dt))
    return [
        [float(times[i]), float(times[i + 1])]
        for i, d in enumerate(dt)
        if d > factor * med
    ]


def integrate(trace, injections, baseline, config=None, budget=10000):
    """Returns (status, payload). Budget exhaustion is a distinct outcome."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    t = np.array([s["t_s"] for s in trace["samples"]], dtype=float)
    p = np.array([s["power_uW"] for s in trace["samples"]], dtype=float)
    bt = np.array([s["t_s"] for s in baseline["samples"]], dtype=float)
    bp = np.array([s["power_uW"] for s in baseline["samples"]], dtype=float)
    gaps = find_gaps(t, cfg["gap_factor"])
    base_at = np.interp(t, bt, bp) if len(bt) else np.zeros_like(t)
    known_times = sorted(inj["time_s"] for inj in injections if inj.get("time_s") is not None)

    payload = {"config": cfg, "gaps": gaps, "injections": [], "diagnostics": []}
    for idx, inj in enumerate(injections):
        if idx + 1 > budget:
            return RESULT_BUDGET, None
        entry = {"injection_id": inj["id"], "index": idx, "status": RESULT_OK,
                 "window": None, "heat_uJ": None, "diagnostics": []}
        t_inj = inj.get("time_s")
        if t_inj is None:
            entry["status"] = "invalid"
            entry["diagnostics"].append(
                diag("missing-time", "injection has no usable time", f"injections/{idx}/time"))
            payload["injections"].append(entry)
            continue

        t0 = t_inj - cfg["pre_s"]
        t1 = t_inj + cfg["post_s"]
        nxt = min((x for x in known_times if x > t_inj), default=None)
        if nxt is not None and t1 > nxt - 0.5:
            t1 = nxt - 0.5
            entry["diagnostics"].append(diag(
                "overlapping-peaks",
                "post-injection peak overlaps next injection rise; window clamped to next injection start",
                f"injections/{idx}/window"))

        status = RESULT_OK
        for g0, g1 in gaps:
            if t0 < g1 and t1 > g0:
                if g0 <= t_inj <= g1:
                    status = "invalid"
                    entry["diagnostics"].append(diag(
                        "gap-at-injection",
                        "injection time falls inside a data gap",
                        f"injections/{idx}/time"))
                    break
                if g0 > t_inj:
                    t1 = min(t1, g0)
                else:
                    t0 = max(t0, g1)
                status = "truncated_gap"
                entry["diagnostics"].append(diag(
                    "window-truncated-at-gap",
                    "integration window truncated at data gap edge; window never crosses a gap",
                    f"injections/{idx}/window"))

        if status != "invalid":
            mask = (t >= t0) & (t <= t1)
            if int(mask.sum()) < 3:
                status = "invalid"
                entry["diagnostics"].append(diag(
                    "insufficient-samples", "fewer than 3 samples inside window",
                    f"injections/{idx}/window"))
            else:
                entry["heat_uJ"] = float(_trapz((p - base_at)[mask], t[mask]))
                entry["window"] = [float(t0), float(t1)]

        if idx == 0 and cfg["discard_first"] and status in (RESULT_OK, "truncated_gap"):
            status = "discarded"
            entry["diagnostics"].append(diag(
                "first-injection-discarded",
                "first injection discarded per integration config; kept on record, excluded from fits",
                f"injections/{idx}"))
        entry["status"] = status
        payload["injections"].append(entry)
    return RESULT_OK, payload
