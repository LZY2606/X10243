"""Per-injection integration of ITC power curves.

Rules encoded here (each is also emitted as a machine-readable event):
- post-injection peaks may overlap: a window ends at the next injection
  start even when the previous peak has not returned to baseline;
- an integration window never crosses a data gap: it is clamped at the gap
  start and the injection is flagged with a located diagnostic;
- the final window ends at the run end (integration endpoint).
"""
import numpy as np

PARAM_VERSION = "itc-intg/1.0"
OVERLAP_TOL_UJ_S = 0.05

_trapz = getattr(np, "trapezoid", np.trapz)


def integrate_run_data(times, power, injections, baseline, gaps, t_end):
    """Integrate each injection peak against a linear baseline.

    times/power: canonical arrays (s, uJ/s). injections: list of dicts with
    index/time_s/volume_uL. baseline: {"intercept","slope"} in uJ/s.
    gaps: list of [t0,t1] sample-free intervals. Returns (results, events).
    """
    times = np.asarray(times, float)
    power = np.asarray(power, float)
    base = baseline["intercept"] + baseline["slope"] * times
    results = []
    events = []
    n = len(injections)
    for k, inj in enumerate(injections):
        idx = int(inj["index"])
        t0 = float(inj["time_s"])
        vol = float(inj["volume_uL"])
        end = float(injections[k + 1]["time_s"]) if k + 1 < n else float(t_end)
        flags = []
        diags = []
        if k + 1 == n:
            flags.append("window_ends_at_run_end")
        if k + 1 < n:
            p_next = float(np.interp(end, times, power))
            b_next = baseline["intercept"] + baseline["slope"] * end
            if p_next - b_next > OVERLAP_TOL_UJ_S:
                flags.append("peaks_overlap")
                events.append({
                    "kind": "integration.peaks_overlap",
                    "payload": {
                        "rule": "post-injection peaks may overlap; window ends at "
                                "next injection start",
                        "injection": idx,
                        "next_injection": int(injections[k + 1]["index"]),
                        "window": [t0, end],
                        "residual_power_uJ_s": round(p_next - b_next, 4),
                    }})
        for g in gaps:
            g0, g1 = float(g[0]), float(g[1])
            if t0 < g0 < end:
                end = g0
                flags.append("truncated_by_gap")
                diags.append({
                    "code": "window_clamped_by_gap",
                    "message": "integration window clamped at data-gap start; "
                               "a window never crosses a gap",
                    "location": f"injection[{idx}]",
                    "gap": [g0, g1]})
                events.append({
                    "kind": "integration.window_clamped",
                    "payload": {
                        "rule": "integration window must not cross data gaps",
                        "injection": idx,
                        "gap": [g0, g1],
                        "clamped_end": g0}})
        mask = (times >= t0) & (times < end)
        if int(mask.sum()) < 2:
            heat = None
            diags.append({
                "code": "insufficient_samples",
                "message": "fewer than 2 power samples inside the window",
                "location": f"injection[{idx}]"})
        else:
            heat = float(_trapz((power - base)[mask], times[mask]))
        results.append({
            "index": idx,
            "time_s": t0,
            "volume_uL": vol,
            "window": [t0, end],
            "heat_uJ": heat,
            "flags": flags,
            "status": "diagnostic" if diags else "ok",
            "diagnostics": diags})
    return results, events
