"""Per-injection integration of ITC power traces.

Domain rules enforced here (pure functions; the pipeline records the
corresponding machine-readable events):

- Post-injection peaks MAY overlap: a window is clamped at the next
  injection time, never extended past it.
- An integration window must NOT cross a data gap: it is truncated at
  the gap start (or invalidated if the gap covers the window start).
- First-injection discard keeps the integral but excludes the point
  from downstream fitting.
"""

from .errors import OK, SEMANTIC_UNSOLVABLE, diag, make_result

DEFAULT_PARAMS = {
    "max_window_s": 120.0,
    "min_points": 3,
    "min_coverage": 0.5,
    "discard_first": False,
}


def baseline_at(baseline, t):
    """Linear interpolation on a baseline curve [(t, p), ...]."""
    if not baseline:
        return 0.0
    if t <= baseline[0][0]:
        return baseline[0][1]
    if t >= baseline[-1][0]:
        return baseline[-1][1]
    lo, hi = 0, len(baseline) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if baseline[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, p0 = baseline[lo]
    t1, p1 = baseline[hi]
    if t1 == t0:
        return p0
    return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


def integrate(power, baseline, injections, gaps, params=None):
    """power/baseline: canonical units (s, uJ/s). injections: [{t, v}].
    gaps: [[t0, t1], ...] declared in the power material."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    diags = []
    if not injections:
        return make_result(SEMANTIC_UNSOLVABLE, None,
                           [diag("injections", "no_injections",
                                 "injection schedule is empty")])
    if not power:
        return make_result(SEMANTIC_UNSOLVABLE, None,
                           [diag("power", "no_power_samples",
                                 "power series is empty")])
    out = []
    n = len(injections)
    for i, inj in enumerate(injections):
        start = float(inj["t"])
        vol = float(inj["v"])
        flags = []
        end = start + p["max_window_s"]
        if i + 1 < n and float(injections[i + 1]["t"]) < end:
            # overlapping post-injection peaks are allowed; clamp window
            end = float(injections[i + 1]["t"])
            flags.append("overlap_clamped")
        gap_hit = None
        for g in gaps:
            g0, g1 = float(g[0]), float(g[1])
            if g0 < end and g1 > start:
                gap_hit = [g0, g1]
                if g0 > start:
                    end = min(end, g0)  # window must not cross the gap
                    flags.append("gap_truncated")
                else:
                    end = start
                    flags.append("gap_covers_start")
        rec = {"index": i, "t": start, "volume_uL": vol,
               "window": [start, end], "flags": flags,
               "status": OK, "heat_uJ": None}
        if gap_hit:
            diags.append(diag(
                f"injections/{i}/window", "gap_intersects_window",
                f"integration window for injection {i} truncated at data gap "
                f"[{gap_hit[0]}, {gap_hit[1]}] s", severity="warning"))
        pts = [(t, y) for (t, y) in power if start <= t <= end]
        span = end - start
        coverage = 0.0
        if len(pts) >= 2 and span > 0:
            coverage = (pts[-1][0] - pts[0][0]) / span
        if end <= start or len(pts) < p["min_points"] or coverage < p["min_coverage"]:
            rec["status"] = "invalid"
            diags.append(diag(
                f"injections/{i}", "integration_impossible",
                f"injection {i}: window [{start}, {end}] s has "
                f"{len(pts)} samples, coverage {coverage:.2f}"))
        else:
            heat = 0.0
            for (t0, y0), (t1, y1) in zip(pts, pts[1:]):
                b0 = baseline_at(baseline, t0)
                b1 = baseline_at(baseline, t1)
                heat += 0.5 * ((y0 - b0) + (y1 - b1)) * (t1 - t0)
            rec["heat_uJ"] = heat
            rec["n_samples"] = len(pts)
            rec["coverage"] = coverage
        if i == 0 and p["discard_first"]:
            # first-injection discard: integral retained, excluded from fits
            rec["flags"].append("discarded")
            rec["status"] = "discarded"
        out.append(rec)
    return make_result(OK, {"injections": out, "params": p}, diags)
