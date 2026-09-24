"""Injection integration engine.

Rules encoded here (mirrored as machine-readable events by the project
layer):

  * Peaks after an injection MAY overlap: when a peak's tail runs into the
    next injection, the window is capped at the next injection start and
    the injection is flagged ``peak_overlap`` - it is still integrated.
  * An integration window MUST NOT cross a data gap: the window is
    truncated at the gap start and a located diagnostic
    ``window_truncated_at_gap`` is attached.  An injection whose start
    lies inside a gap cannot be integrated at all (``injection_in_gap``);
    no heat value is fabricated for it.
"""

from __future__ import annotations

import numpy as np

PARAM_VERSION = "integration/1"


def _invalid(code, location, message):
    return {"status": "invalid_format", "injections": [],
            "diagnostics": [{"code": code, "location": location,
                             "message": message}]}


def _settle_end(t, resid, t0, window_max, settle_tol, settle_span):
    """First time after t0 where |residual| stays below settle_tol for
    settle_span seconds; otherwise t0 + window_max."""
    end_limit = t0 + window_max
    idx = int(np.searchsorted(t, t0))
    n = t.size
    j = idx
    while j < n and t[j] <= end_limit:
        k = int(np.searchsorted(t, t[j] + settle_span))
        if k >= n or t[min(k, n - 1)] > end_limit:
            break
        if bool(np.all(np.abs(resid[j:k]) < settle_tol)):
            return float(t[j])
        j += 1
    return float(end_limit)


def integrate(power, baseline, injections, gaps,
              window_max=180.0, settle_tol=0.05, settle_span=20.0):
    """Integrate each injection peak against the baseline curve.

    power/baseline: {"t": [...], "p": [...]} in seconds and uJ/s.
    injections: [{"t": seconds, "volume": uL}, ...]
    gaps: [[t0, t1], ...] ranges with no valid samples.
    """
    try:
        t = np.asarray(power["t"], dtype=float)
        p = np.asarray(power["p"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid("power_curve_shape", "power_curve",
                        f"power curve must provide numeric 't' and 'p' "
                        f"arrays ({exc})")
    if t.ndim != 1 or p.shape != t.shape or t.size < 2:
        return _invalid("power_curve_shape", "power_curve",
                        "'t' and 'p' must be equal-length 1-D arrays with "
                        "at least 2 samples")
    if not bool(np.all(np.diff(t) > 0)):
        return _invalid("power_curve_order", "power_curve.t",
                        "sample times must be strictly increasing")
    try:
        bt = np.asarray(baseline["t"], dtype=float)
        bp = np.asarray(baseline["p"], dtype=float)
        base = np.interp(t, bt, bp)
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid("baseline_shape", "baseline_curve",
                        f"baseline curve must provide numeric 't' and 'p' "
                        f"arrays ({exc})")

    resid = p - base
    times = [float(inj["t"]) for inj in injections]
    results = []
    diagnostics = []
    for i, inj in enumerate(injections):
        t0 = float(inj["t"])
        vol = float(inj["volume"])
        loc = f"injection[{i + 1}]"
        flags: list[str] = []
        diags: list[dict] = []

        in_gap = next((g for g in gaps if g[0] <= t0 <= g[1]), None)
        if in_gap is not None:
            flags.append("in_gap")
            diags.append({
                "code": "injection_in_gap", "location": loc,
                "gap": list(in_gap),
                "message": "injection starts inside a data gap; no heat "
                           "value is fabricated"})
            results.append({"index": i + 1, "t": t0, "volume": vol,
                            "window": None, "heat": None, "flags": flags,
                            "diagnostics": diags})
            diagnostics.extend(diags)
            continue

        end = _settle_end(t, resid, t0, window_max, settle_tol, settle_span)
        nxt = times[i + 1] if i + 1 < len(times) else None
        if nxt is not None and end > nxt:
            flags.append("peak_overlap")
            end = nxt
        for g in gaps:
            if t0 < g[0] and end > g[0]:
                end = min(end, float(g[0]))
                flags.append("gap_truncated")
                diags.append({
                    "code": "window_truncated_at_gap", "location": loc,
                    "gap": [float(g[0]), float(g[1])],
                    "window_end": end,
                    "message": "integration window must not cross a data "
                               "gap; truncated at gap start"})

        mask = (t >= t0) & (t <= end)
        if int(mask.sum()) < 2:
            diags.append({
                "code": "insufficient_samples", "location": loc,
                "window": [t0, end],
                "message": "fewer than 2 samples inside the integration "
                           "window"})
            results.append({"index": i + 1, "t": t0, "volume": vol,
                            "window": [t0, end], "heat": None,
                            "flags": flags, "diagnostics": diags})
            diagnostics.extend(diags)
            continue

        heat = float(np.trapezoid(resid[mask], t[mask]))
        results.append({"index": i + 1, "t": t0, "volume": vol,
                        "window": [t0, end], "heat": heat,
                        "flags": flags, "diagnostics": diags})
        diagnostics.extend(diags)

    status = "ok" if all(r["heat"] is not None for r in results) \
        else "incomplete_input"
    return {"status": status, "injections": results,
            "diagnostics": diagnostics,
            "params": {"window_max": window_max, "settle_tol": settle_tol,
                       "settle_span": settle_span}}
