"""Per-injection integration of the power curve.

Hard rules (these are the tricky cases):
  * post-injection peaks are allowed to overlap: a window ends at the start of
    the next injection, never by silently stealing the neighbour's peak;
  * an integration window must never cross a data gap. A nominal window that
    spans a gap is ``crosses_gap`` (not a number pretending to be valid); an
    explicit reviewer endpoint that lands before the gap is a documented
    ``manual_endpoint`` clipping;
  * integration endpoints and baseline reference are part of the payload.
"""
from __future__ import annotations

import numpy as np

from .errors import Diagnostic
from .models import make_node

DEFAULT_INTEGRATION = {
    "version": 1,
    "window_s": 200.0,
    "min_points": 8,
    "min_coverage_fraction": 0.30,
    "endpoints": "next_injection",   # or "fixed"
    "baseline_mode": "material_linear",
    "time_reference": "s after acquisition start",
}

STATUS_OK = "integrated"
STATUS_OVERLAP = "integrated_overlap_clip"
STATUS_GAP = "crosses_gap"
STATUS_CLIPPED = "manual_endpoint"
STATUS_SHORT = "insufficient_coverage"
STATUS_OUTSIDE = "outside_trace"


def _segment_gap(t0: float, t1: float, gaps: list[dict]) -> dict | None:
    for gap in gaps:
        if gap["from_s"] >= t0 and gap["to_s"] <= t1:
            return gap
    return None


def integrate_injection(power: dict, schedule: dict, baseline: dict | None,
                        params: dict, ordinal: int,
                        manual_end_s: float | None = None,
                        discarded: bool = False):
    """Integrate one injection. Returns a Node (valid or structured invalid)."""
    injections = schedule["injections"]
    inj = injections[ordinal]
    t_start = inj["time_s"]
    if t_start is None:
        return _invalid_node(power, schedule, baseline, params, ordinal,
                             manual_end_s, "time_unusable",
                             "injection time has no normalized value",
                             inj.get("time_raw"))

    times = np.asarray(power["time_s"], dtype=float)
    powers = np.asarray(
        [np.nan if p is None else p for p in power["power_cal_s"]],
        dtype=float)

    if t_start < times[0] or t_start >= times[-1]:
        return _invalid_node(power, schedule, baseline, params, ordinal,
                             manual_end_s, STATUS_OUTSIDE,
                             "injection time is outside sampled power range",
                             inj.get("time_raw"))

    # Nominal end: next injection (overlap allowed) or fixed window.
    fixed_end = t_start + float(params["window_s"])
    next_start = (injections[ordinal + 1]["time_s"]
                  if ordinal + 1 < len(injections) else None)
    if params.get("endpoints", "next_injection") == "next_injection" \
            and next_start is not None and next_start > t_start:
        nominal_end = min(fixed_end, next_start)
        overlap_clipped = fixed_end > next_start
    else:
        nominal_end = fixed_end
        overlap_clipped = False
    nominal_end = min(nominal_end, float(times[-1]))

    manual = manual_end_s is not None
    end = float(manual_end_s) if manual else nominal_end
    if end <= t_start:
        return _invalid_node(power, schedule, baseline, params, ordinal,
                             manual_end_s, "endpoint_before_start",
                             "integration end is not after injection time",
                             {"end_s": end, "start_s": t_start})

    gap = _segment_gap(t_start, end, power.get("gaps", []))
    if gap is not None:
        return _invalid_node(power, schedule, baseline, params, ordinal,
                             manual_end_s, STATUS_GAP,
                             "integration window spans a data gap",
                             {"gap": gap, "window": [t_start, end]})

    mask = (times >= t_start) & (times <= end)
    seg_t = times[mask]
    seg_p = powers[mask]
    coverage = (seg_t[-1] - seg_t[0]) if seg_t.size else 0.0
    nominal_span = max(end - t_start, 1e-12)
    enough_points = seg_t.size >= int(params["min_points"])
    enough_coverage = coverage / nominal_span >= float(
        params["min_coverage_fraction"])
    if seg_t.size < 2 or not enough_points or not enough_coverage:
        return _invalid_node(
            power, schedule, baseline, params, ordinal, manual_end_s,
            STATUS_SHORT,
            f"only {seg_t.size} samples / "
            f"{coverage / nominal_span:.0%} coverage inside window",
            {"samples": int(seg_t.size),
             "coverage_fraction": coverage / nominal_span})

    signal = seg_p.copy()
    if baseline is not None and baseline.get("baseline_cal_s") is not None:
        bt = np.asarray(baseline["time_s"], dtype=float)
        bv = np.asarray(
            [np.nan if v is None else v
             for v in baseline["baseline_cal_s"]], dtype=float)
        good = ~np.isnan(bt) & ~np.isnan(bv)
        if good.sum() >= 2:
            signal = signal - np.interp(seg_t, bt[good], bv[good])
        baseline_mode = "material"
    else:
        baseline_mode = "linear_window_endpoints"

    if baseline_mode == "linear_window_endpoints":
        line = signal[0] + (signal[-1] - signal[0]) * \
            (seg_t - seg_t[0]) / (seg_t[-1] - seg_t[0])
        resid = signal - line
    else:
        resid = signal

    heat_cal = float(np.trapezoid(resid, seg_t))
    if manual:
        status = STATUS_CLIPPED
    elif overlap_clipped:
        status = STATUS_OVERLAP
    else:
        status = STATUS_OK

    diagnostics = []
    if overlap_clipped and not manual:
        diagnostics.append(Diagnostic(
            "overlap_window_clipped",
            f"injection #{ordinal} peak overlaps next injection; window ends "
            "at next injection time", "info",
            f"schedule:injection:{ordinal}",
            {"fixed_end_s": fixed_end, "next_injection_s": next_start}))
    if manual:
        diagnostics.append(Diagnostic(
            "manual_endpoint",
            f"injection #{ordinal} integrated to reviewer endpoint {end:g}s "
            "(nominal window not crossed-gap)", "info",
            f"schedule:injection:{ordinal}", {"end_s": end}))

    payload = {
        "ordinal": ordinal,
        "label": inj["label"],
        "status": status,
        "discarded": bool(discarded),
        "heat_cal": heat_cal,
        "heat_unit": "cal",
        "t_start_s": t_start,
        "t_end_s": float(end),
        "nominal_end_s": float(nominal_end),
        "fixed_end_s": float(fixed_end),
        "next_injection_s": next_start,
        "overlap_clipped": bool(overlap_clipped),
        "manual_endpoint": manual,
        "baseline_mode": baseline_mode,
        "time_reference": "s after acquisition start",
        "samples": int(seg_t.size),
        "volume_raw": inj["volume_raw"],
        "gap_after": [g for g in power.get("gaps", [])
                      if g["from_s"] >= t_start and g["to_s"] <= end],
    }
    inputs = {
        "power": power_id_basis(power),
        "schedule": schedule_id_basis(schedule),
        "baseline": (stable_short(baseline) if baseline else None),
        "ordinal": ordinal,
        "manual_end_s": manual_end_s,
        "discarded": bool(discarded),
    }
    return make_node("integration", status, payload, inputs, [],
                     {"integration": int(params["version"])}, diagnostics)


def power_id_basis(power: dict) -> dict:
    return {
        "median_dt_s": power["median_dt_s"],
        "gap_after": power["gap_after"],
        "n": len(power["time_s"]),
        "checksum": _checksum(power["power_cal_s"]),
        "t0": power["time_s"][0],
        "tn": power["time_s"][-1],
    }


def schedule_id_basis(schedule: dict) -> dict:
    return {"injections": schedule["injections"]}


def stable_short(payload: dict) -> str:
    from .models import stable_hash
    return stable_hash(payload)


def _checksum(values) -> str:
    arr = np.asarray(
        [np.nan if v is None else v for v in values], dtype=float)
    return f"{float(np.nansum(arr)):.9g}:{int(arr.size)}"


def _invalid_node(power, schedule, baseline, params, ordinal, manual_end_s,
                  status, message, evidence) :
    diagnostics = [Diagnostic(
        f"integration_{status}", message, "error",
        f"schedule:injection:{ordinal}", evidence,
        context={"ordinal": ordinal, "window_rule":
                 params.get("endpoints", "next_injection")})]
    inj = schedule["injections"][ordinal]
    payload = {
        "ordinal": ordinal, "label": inj["label"], "status": status,
        "discarded": False, "heat_cal": None, "heat_unit": "cal",
        "t_start_s": inj["time_s"], "t_end_s": None,
        "manual_endpoint": manual_end_s is not None,
        "time_reference": "s after acquisition start",
        "failure": message, "evidence": _jsonable(evidence),
        "volume_raw": inj.get("volume_raw"),
    }
    inputs = {
        "power": power_id_basis(power),
        "schedule": schedule_id_basis(schedule),
        "baseline": (stable_short(baseline) if baseline else None),
        "ordinal": ordinal, "manual_end_s": manual_end_s,
    }
    return make_node("integration", status, payload, inputs, [],
                     {"integration": int(params["version"])}, diagnostics)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
