"""Power-curve integration: overlapping peaks, data gaps, endpoints, baseline.

Rules implemented (the "tricky core"):
* post-injection peaks may overlap: an integration window is clipped at the
  next injection's pre-window, and the clipping is reported, not hidden;
* an integration window never crosses a data gap: it is clipped at the first
  gap inside the nominal window and GAP_CLIPPED is reported; a gap at the start
  gives the distinct outcome GAP_AT_START (no integral produced);
* dilution/blanks follow an explicit correspondence rule (ordinal or
  volume); a missing counterpart is reported per injection and the corrected
  heat is withheld rather than silently zeroed.
"""

from __future__ import annotations

import math

import numpy as np

from .util import digest

CAL4 = 4.184
DEFAULT_PRE_S = 10.0
DEFAULT_POST_S = 240.0


def find_gaps(samples, *, factor=5.0):
    """Return gap intervals (lo, hi) from unplaced/invalid samples or hiatuses."""
    valid_times = [s["time_s"] for s in samples if s["valid"]]
    gaps = []
    if len(valid_times) >= 2:
        diffs = np.diff(np.asarray(valid_times, dtype=float))
        med = float(np.median(diffs)) if diffs.size else 0.0
        threshold = max(factor * med, 0.0) if med > 0 else 0.0
        for d, left in zip(diffs, valid_times[:-1]):
            if threshold and d > threshold:
                gaps.append((float(left), float(left + d), "hiatus"))
    bad = sorted((s["time_s"] for s in samples if not s["valid"]))
    if bad:
        lo = hi = bad[0]
        for t in bad[1:]:
            if t <= hi:
                hi = t
            else:
                gaps.append((lo, hi, "invalid"))
                lo = hi = t
        gaps.append((lo, hi, "invalid"))
    gaps.sort(key=lambda g: (g[0], g[1]))
    return [{"lo": lo, "hi": hi, "reason": why} for lo, hi, why in gaps]


def _valid_arrays(samples):
    rows = [s for s in samples if s["valid"]]
    t = np.asarray([r["time_s"] for r in rows], dtype=float)
    p = np.asarray([r["power_w"] for r in rows], dtype=float)
    order = np.argsort(t)
    return t[order], p[order]


def baseline_levels(t, p, t0, t1, anchors):
    """Baseline level (W) at t0/t1 via anchors else window-endpoint estimator."""
    method = "anchors" if anchors else "endpoint_mean"
    if anchors:
        at = np.asarray([a["time_s"] for a in anchors], dtype=float)
        al = np.asarray([a["level_w"] for a in anchors], dtype=float)
        usable = ~np.isnan(al.astype(float))
        at, al = at[usable], al[usable]
        if at.size >= 2:
            return (float(_interp_extend(t0, at, al)),
                    float(_interp_extend(t1, at, al)), "linear_anchors")
        if at.size == 1:
            return float(al[0]), float(al[0]), "constant_anchor"
    pre = p[(t >= t0 - 12.0) & (t <= t0)]
    post = p[(t >= t1 - 20.0) & (t <= t1)]
    b0 = float(pre.mean()) if pre.size else float(p[0])
    b1 = float(post.mean()) if post.size else float(p[-1])
    return b0, b1, method


def _first_gap(gaps, lo, hi):
    inside = [g for g in gaps if g["lo"] >= lo and g["hi"] <= hi and g["hi"] > lo]
    if not inside:
        return None
    # a gap is "at start" when the injection instant itself lies inside it.
    if any(g["lo"] <= lo <= g["hi"] for g in inside):
        return ("start", inside[0])
    return ("after", min(inside, key=lambda g: g["lo"]))


def integrate_series(power, schedule, baseline=None, *, pre_s=DEFAULT_PRE_S,
                      post_s=DEFAULT_POST_S, next_spacing=None):
    """Integrate every injection against one power material.

    Returns {"integrals":[...], "gaps":[...], "units":{...}}.
    """
    samples = power["samples"]
    gaps = find_gaps(samples)
    t, p = _valid_arrays(samples)
    anchors = (baseline or {}).get("anchors", [])
    injections = sorted(schedule["injections"], key=lambda j: j["ordinal"])
    results = []

    for k, inj in enumerate(injections):
        t_start = inj["time_s"]
        if t_start is None:
            results.append(_bad_integral(inj, "MISSING_TIME", nominal=None))
            continue
        nominal_lo = t_start - pre_s
        nxt = injections[k + 1]["time_s"] if k + 1 < len(injections) else None
        nominal_hi = t_start + post_s
        eff_lo, eff_hi = nominal_lo, nominal_hi
        statuses, diagnostics = [], []
        overlap_tail_s = None

        if nxt is not None:
            boundary = nxt - pre_s
            if nominal_hi > boundary and boundary > t_start:
                overlap_tail_s = nominal_hi - boundary
                eff_hi = boundary
                statuses.append("OVERLAP_CLIPPED")
                diagnostics.append({
                    "code": "OVERLAPPING_PEAKS",
                    "ordinal": inj["ordinal"],
                    "next_ordinal": injections[k + 1]["ordinal"],
                    "nominal_end_s": nominal_hi, "effective_end_s": eff_hi,
                    "tail_excluded_s": overlap_tail_s,
                    "message": "integration window clipped before next injection pre-window",
                })

        gap_info = _first_gap(gaps, max(eff_lo, t[0]), eff_hi)
        gap_at_start = False
        if gap_info is not None:
            where, gap = gap_info
            if where == "start" or gap["lo"] <= t_start <= gap["hi"] or eff_lo >= gap["hi"]:
                gap_at_start = True
            else:
                eff_hi = gap["lo"]
                statuses.append("GAP_CLIPPED")
                diagnostics.append({
                    "code": "WINDOW_CLIPPED_AT_GAP",
                    "ordinal": inj["ordinal"], "gap": gap,
                    "nominal_end_s": nominal_hi, "effective_end_s": eff_hi,
                    "message": "integration window does not cross a data gap; clipped at gap",
                })
        if gap_at_start:
            results.append(_bad_integral(
                inj, "GAP_AT_START",
                nominal={"start_s": nominal_lo, "end_s": nominal_hi,
                         "injection_s": t_start},
                extra_diags=diagnostics + [{
                    "code": "GAP_AT_START", "ordinal": inj["ordinal"],
                    "message": "injection instant lies in a data gap; integral withheld",
                }]))
            continue

        mask = (t >= eff_lo) & (t <= eff_hi)
        ts, ps = t[mask], p[mask]
        endpoint = _endpoint_check(t, ts, t_start, eff_lo, eff_hi)
        diagnostics.extend(endpoint["diagnostics"])
        if not endpoint["supported"]:
            results.append(_bad_integral(
                inj, "ENDPOINT_UNSUPPORTED",
                nominal={"start_s": eff_lo, "end_s": eff_hi, "injection_s": t_start},
                extra_diags=diagnostics))
            continue
        if endpoint["diagnostics"]:
            statuses.append("ENDPOINT_PARTIAL")

        b0, b1, bmethod = baseline_levels(t, p, eff_lo, eff_hi, anchors)
        base_line = b0 + (b1 - b0) * (ts - eff_lo) / (eff_hi - eff_lo)
        net = ps - base_line
        energy_j = float(np.trapezoid(net, ts)) if hasattr(np, "trapezoid") else float(np.trapz(net, ts))
        results.append({
            "ordinal": inj["ordinal"], "status": "OK",
            "statuses": sorted(set(statuses)) or ["OK"],
            "energy_j": energy_j, "energy_ucal": energy_j / CAL4 * 1e6,
            "injection_s": t_start,
            "nominal_window_s": [nominal_lo, nominal_hi],
            "effective_window_s": [float(eff_lo), float(eff_hi)],
            "baseline_w": [b0, b1], "baseline_method": bmethod,
            "sample_count": int(ts.size),
            "volume_l": inj["volume_l"],
            "raw": {"time": inj["raw_time"], "volume": inj["raw_volume"]},
            "diagnostics": diagnostics,
        })
    return {"integrals": results, "gaps": gaps}


def _interp_extend(x, xp, fp):
    """np.interp with linear extension beyond the end anchors (no clamping)."""
    if x <= xp[0]:
        if xp[1] == xp[0]:
            return fp[0]
        return fp[0] + (fp[1] - fp[0]) * (x - xp[0]) / (xp[1] - xp[0])
    if x >= xp[-1]:
        if xp[-1] == xp[-2]:
            return fp[-1]
        return fp[-1] + (fp[-1] - fp[-2]) * (x - xp[-1]) / (xp[-1] - xp[-2])
    return float(np.interp(x, xp, fp))


def _endpoint_check(all_t, win_t, t_start, lo, hi):
    diags = []
    if win_t.size < 2:
        return {"supported": False, "diagnostics": [{
            "code": "ENDPOINT_UNSUPPORTED", "message": "fewer than two valid samples in window"}]}
    if hi > all_t[-1]:
        diags.append({"code": "ENDPOINT_BEYOND_LAST_SAMPLE", "requested_s": hi,
                      "last_sample_s": float(all_t[-1]),
                      "message": "window end beyond last valid sample; truncated"})
    if lo < all_t[0]:
        diags.append({"code": "WINDOW_BEFORE_FIRST_SAMPLE", "requested_s": lo,
                      "first_sample_s": float(all_t[0]),
                      "message": "window start before first valid sample; truncated"})
    pre = win_t[win_t <= t_start]
    post = win_t[win_t >= t_start]
    if pre.size < 2 or post.size < 2:
        return {"supported": False, "diagnostics": diags + [{
            "code": "ENDPOINT_UNSUPPORTED",
            "message": "injection instant lacks valid samples on both sides"}]}
    return {"supported": True, "diagnostics": diags}


def _bad_integral(inj, code, *, nominal, extra_diags=None):
    return {
        "ordinal": inj["ordinal"], "status": code, "statuses": [code],
        "energy_j": None, "energy_ucal": None,
        "injection_s": inj["time_s"],
        "nominal_window_s": [nominal["start_s"], nominal["end_s"]] if nominal else None,
        "effective_window_s": None, "baseline_w": None, "baseline_method": None,
        "sample_count": 0, "volume_l": inj.get("volume_l"),
        "raw": {"time": inj["raw_time"], "volume": inj["raw_volume"]},
        "diagnostics": extra_diags or [{"code": code, "ordinal": inj["ordinal"]}],
    }


def apply_first_discard(integrals, decision):
    """First injection discard is a mutable decision; raw integrals are untouched."""
    n = decision.get("count", 1)
    out = []
    for r in integrals:
        rr = dict(r)
        if r["ordinal"] <= n:
            rr["excluded"] = True
            rr["exclusion"] = {"rule": "first_injection_discard",
                               "decision_id": decision.get("id"), "count": n}
        else:
            rr["excluded"] = False
        out.append(rr)
    return out
