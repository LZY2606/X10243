"""Injection integration and dilution-heat (blank) correction.

Hard rules implemented here (each emits a machine-readable judgement):
  * post-injection peaks MAY overlap; windows split at the next injection
  * an integration window never crosses a data gap (truncated at gap edge)
  * dilution-heat subtraction requires an explicit correspondence rule;
    missing blank counterparts stay uncorrected (never zero-filled)
"""
from __future__ import annotations

PARAM_VERSION = "integrate-1.0"

DEFAULT_PARAMS = {
    "pre_window_s": 2.0,
    "post_window_s": 60.0,
    "overlap_threshold_s": 45.0,
    "gap_rule": "truncate_at_gap",
    "overlap_rule": "split_at_next_injection",
    "blank_correspondence": "by_injection_index",
}


def _baseline_at(baseline, t):
    kind = baseline.get("kind", "constant")
    coeff = baseline["coefficients"]
    if kind == "constant":
        return coeff[0]
    if kind == "linear":
        return coeff[0] + coeff[1] * (t - baseline.get("t0", 0.0))
    raise ValueError(f"unknown baseline kind {kind!r}")


def integrate_injections(power, injections, baseline, params=None,
                         diagnostics=None):
    """power: {'time':[s], 'power':[uJ/s], 'gaps':[{start_s,end_s,reason}]}
    injections: [{'index','time_s','volume_uL'}]
    Returns (results, judgements)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    diagnostics = diagnostics if diagnostics is not None else []
    judgements = []
    t, y = power["time"], power["power"]
    gaps = sorted(power.get("gaps", []), key=lambda g: g["start_s"])
    results = []
    for i, inj in enumerate(injections):
        pos = f"injections[{i}]"
        t0 = inj["time_s"]
        if t0 is None or inj.get("volume_uL") is None:
            diagnostics.append({
                "severity": "error", "position": pos,
                "code": "injection_incomplete",
                "message": "injection time or volume unconvertible; "
                           "left unintegrated"})
            results.append({"index": i, "status": "invalid_input",
                            "heat_uJ": None, "window_s": None,
                            "overlap": False, "truncated_by_gap": False})
            continue
        start = t0
        end = t0 + p["post_window_s"]
        overlap = False
        if i + 1 < len(injections) and injections[i + 1]["time_s"] is not None:
            nxt = injections[i + 1]["time_s"]
            if nxt - t0 < p["overlap_threshold_s"]:
                overlap = True
                judgements.append({
                    "rule": "post_injection_peaks_may_overlap",
                    "injections": [i, i + 1],
                    "detail": f"spacing {nxt - t0:.3g}s < threshold "
                              f"{p['overlap_threshold_s']}s; window of "
                              f"injection {i} ends at next injection",
                })
            end = min(end, nxt)
        truncated = False
        for g in gaps:
            if g["start_s"] > start and g["start_s"] < end:
                end = g["start_s"]
                truncated = True
                judgements.append({
                    "rule": "integration_window_never_crosses_data_gap",
                    "injection": i,
                    "gap": g,
                    "detail": f"window truncated at gap start "
                              f"{g['start_s']}s ({g.get('reason','')})",
                })
                break
        idx = [k for k in range(len(t))
               if t[k] is not None and start <= t[k] <= end]
        if len(idx) < 2:
            diagnostics.append({
                "severity": "error", "position": pos,
                "code": "empty_integration_window",
                "message": f"fewer than 2 power samples in window "
                           f"[{start}, {end}] s"})
            results.append({"index": i, "status": "empty_window",
                            "heat_uJ": None, "window_s": [start, end],
                            "overlap": overlap, "truncated_by_gap": truncated})
            continue
        heat = 0.0
        for a, b in zip(idx, idx[1:]):
            fa = y[a] - _baseline_at(baseline, t[a])
            fb = y[b] - _baseline_at(baseline, t[b])
            heat += 0.5 * (fa + fb) * (t[b] - t[a])
        status = "ok"
        if truncated:
            status = "truncated_by_gap"
        elif overlap:
            status = "ok_overlap"
        results.append({
            "index": i, "status": status, "heat_uJ": heat,
            "window_s": [start, end], "n_samples": len(idx),
            "overlap": overlap, "truncated_by_gap": truncated,
            "volume_uL": inj["volume_uL"],
        })
    return results, judgements


def blank_correct(integration_results, blank_heats, params=None,
                  diagnostics=None):
    """Subtract dilution heat using an explicit correspondence rule.
    Missing counterparts are reported, never zero-filled."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    diagnostics = diagnostics if diagnostics is not None else []
    judgements = [{
        "rule": "dilution_correction_requires_correspondence",
        "correspondence": p["blank_correspondence"],
        "detail": "blank heats matched to sample injections by injection "
                  "index; injections without a blank counterpart keep "
                  "uncorrected heat and a diagnostic",
    }]
    out = []
    for r in integration_results:
        rec = dict(r)
        if r["heat_uJ"] is None:
            rec.update(corrected_heat_uJ=None, blank_status="not_applicable")
            out.append(rec)
            continue
        j = r["index"] if p["blank_correspondence"] == "by_injection_index" \
            else None
        if j is None or j >= len(blank_heats) or blank_heats[j] is None:
            diagnostics.append({
                "severity": "warning",
                "position": f"injections[{r['index']}]",
                "code": "blank_counterpart_missing",
                "message": f"no blank heat for injection {r['index']} under "
                           f"rule {p['blank_correspondence']}; heat left "
                           "uncorrected"})
            rec.update(corrected_heat_uJ=None, blank_status="missing")
        else:
            rec.update(
                corrected_heat_uJ=r["heat_uJ"] - blank_heats[j],
                blank_heat_uJ=blank_heats[j], blank_status="subtracted")
        out.append(rec)
    return out, judgements
