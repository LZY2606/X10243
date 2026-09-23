"""Injection integration engine.

Rules encoded here (machine-checkable):
  * post-injection peaks MAY overlap; windows are fixed-length and may
    overlap each other;
  * an integration window must NOT cross a data gap -- if it does, the
    integral is invalid and a located diagnostic is emitted (no value is
    fabricated);
  * window endpoints follow an explicit inclusive/exclusive rule with
    linear interpolation at both ends;
  * the first injection can be discarded by policy (common ITC practice).
"""

import numpy as np

from .errors import BudgetExhausted

DEFAULT_PARAMS = {
    "window_s": 120.0,
    "baseline_window_s": 30.0,
    "endpoint": "inclusive",   # or "exclusive"
    "discard_first": False,
    "max_injections": 10000,   # resource budget
}


def detect_gaps(t, power, max_sample_gap_s=None):
    """Gaps come from explicit null runs and (optionally) sampling holes."""
    gaps = []
    t = np.asarray(t, dtype=float)
    p = np.asarray(power, dtype=object)
    i = 0
    n = len(t)
    while i < n:
        if p[i] is None or (isinstance(p[i], float) and np.isnan(p[i])):
            j = i
            while j + 1 < n and (p[j + 1] is None or
                                 (isinstance(p[j + 1], float) and np.isnan(p[j + 1]))):
                j += 1
            t0 = t[i - 1] if i > 0 else t[i]
            t1 = t[j + 1] if j + 1 < n else t[j]
            gaps.append([float(t0), float(t1)])
            i = j + 1
        else:
            i += 1
    if max_sample_gap_s is not None:
        for k in range(1, n):
            if t[k] - t[k - 1] > max_sample_gap_s:
                gaps.append([float(t[k - 1]), float(t[k])])
    gaps.sort()
    return gaps


def _window_crosses_gap(t0, t1, gaps):
    for g0, g1 in gaps:
        if g0 < t1 and g1 > t0:
            return [g0, g1]
    return None


def _interp(t, p, x):
    return float(np.interp(x, t, p))


def integrate_injections(power, schedule, params=None):
    """Integrate each injection peak.

    `power`: {"t": [...], "power": [...], "gaps": [[t0,t1],...]} (canonical
    units: s, uJ/s). `schedule`: list of {"index","time_s","volume_uL"}.
    Returns {"injections": [...], "diagnostics": [...], "params": ...}.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    injections = sorted(schedule, key=lambda r: (r["time_s"], r["index"]))
    if len(injections) > p["max_injections"]:
        raise BudgetExhausted(
            f"injection count {len(injections)} exceeds budget "
            f"{p['max_injections']}",
            diagnostics=[{"code": "E_BUDGET_INJECTIONS",
                          "message": "too many injections",
                          "location": {"budget": "max_injections"},
                          "severity": "error"}])

    t = np.asarray(power["t"], dtype=float)
    raw_p = power["power"]
    mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v))
                     for v in raw_p])
    pw = np.asarray([v for v, m in zip(raw_p, mask) if m], dtype=float)
    tv = t[mask]
    gaps = [list(g) for g in power.get("gaps", [])]

    out, diags = [], []
    for pos, inj in enumerate(injections):
        iid = f"inj-{inj['index']}"
        rec = {"id": iid, "index": inj["index"], "time_s": inj["time_s"],
               "volume_uL": inj["volume_uL"], "status": "ok",
               "heat_uJ": None, "baseline_uJ_s": None,
               "window": None, "n_samples": 0, "flags": []}
        if p["discard_first"] and pos == 0:
            rec["status"] = "discarded"
            rec["flags"].append("first_injection")
            out.append(rec)
            continue

        t0 = float(inj["time_s"])
        t1 = t0 + float(p["window_s"])
        rec["window"] = [t0, t1]

        gap = _window_crosses_gap(t0, t1, gaps)
        if gap is not None:
            rec["status"] = "invalid"
            rec["flags"].append("window_crosses_gap")
            diags.append({
                "code": "E_WINDOW_CROSSES_GAP",
                "message": f"integration window [{t0},{t1}] crosses data gap {gap}",
                "location": {"injection": iid, "window": [t0, t1], "gap": gap},
                "severity": "error"})
            out.append(rec)
            continue

        b0, b1 = t0 - float(p["baseline_window_s"]), t0
        bsel = (tv >= b0) & (tv < b1)
        if not bsel.any():
            rec["status"] = "invalid"
            rec["flags"].append("baseline_missing")
            diags.append({
                "code": "E_BASELINE_MISSING",
                "message": f"no baseline samples in [{b0},{b1})",
                "location": {"injection": iid, "baseline_window": [b0, b1]},
                "severity": "error"})
            out.append(rec)
            continue
        baseline = float(np.mean(pw[bsel]))
        rec["baseline_uJ_s"] = baseline

        if p["endpoint"] == "inclusive":
            sel = (tv >= t0) & (tv <= t1)
        else:
            sel = (tv >= t0) & (tv < t1)
        xs = list(tv[sel])
        ys = list(pw[sel] - baseline)
        # explicit endpoint handling: interpolate power at both window ends
        xs = [t0] + xs + [t1]
        ys = [_interp(tv, pw, t0) - baseline] + ys + \
             [_interp(tv, pw, t1) - baseline]
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]
        rec["n_samples"] = int(sel.sum())
        rec["heat_uJ"] = float(np.trapz(ys, xs))
        out.append(rec)

    return {"injections": out, "diagnostics": diags, "params": p,
            "gaps": gaps}
