from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import RULES_VERSION
from .errors import SemanticImpossible, diagnostic
from .units import quantity

INTEGRATION_STATUS = {"ok", "warning", "blocked"}


@dataclass
class Trace:
    times: np.ndarray
    values: np.ndarray
    time_unit: str
    value_unit: str


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:16]


def canonical_json_bytes(value):
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def ordered(value: Any, key="id") -> list:
    if isinstance(value, dict):
        items = list(value.values())
    else:
        items = list(value)
    return sorted(items, key=lambda item: (item.get(key, ""), item.get("version", 0)))


def parse_trace(points, path, time_kind="time", value_kind="power"):
    diags = []
    if not isinstance(points, list) or len(points) < 2:
        return None, [diagnostic(path, "trace requires at least two samples", points, "trace_too_short")]
    times = []
    vals = []
    time_unit = None
    value_unit = None
    previous = None
    for idx, point in enumerate(points):
        ppath = f"{path}[{idx}]"
        if not isinstance(point, dict) or "t" not in point or "p" not in point:
            diags.append(diagnostic(ppath, "sample must contain t and p", point, "sample_shape"))
            continue
        t, td = quantity(point["t"], time_kind, ppath + ".t")
        p, pd = quantity(point["p"], value_kind, ppath + ".p")
        diags.extend(td)
        diags.extend(pd)
        if t and p:
            if time_unit is None:
                time_unit = t["canonical_unit"]
            if value_unit is None:
                value_unit = p["canonical_unit"]
            if t["canonical_unit"] != time_unit or p["canonical_unit"] != value_unit:
                diags.append(diagnostic(ppath, "mixed units in one trace", point, "mixed_units"))
                continue
            tv, pv = t["canonical_value"], p["canonical_value"]
            if previous is not None and tv <= previous:
                diags.append(diagnostic(ppath + ".t", "samples must be strictly increasing", tv, "non_monotonic_time"))
                continue
            previous = tv
            times.append(tv)
            vals.append(pv)
    if len(times) < 2:
        return None, diags or [diagnostic(path, "no valid samples", points, "no_valid_samples")]
    return Trace(np.asarray(times, dtype=float), np.asarray(vals, dtype=float), time_unit, value_unit), diags


def qdict(raw, kind, path):
    return quantity(raw, kind, path)


def interpolate(trace: Trace, t: float):
    if t < trace.times[0] or t > trace.times[-1]:
        return None, "outside_trace"
    idx = int(np.searchsorted(trace.times, t, side="left"))
    if idx < len(trace.times) and trace.times[idx] == t:
        return float(trace.values[idx]), None
    if idx == 0 or idx >= len(trace.times):
        return None, "endpoint_unbracketed"
    t0, t1 = trace.times[idx - 1], trace.times[idx]
    if not np.isfinite(t1 - t0) or t1 == t0:
        return None, "endpoint_unbracketed"
    p0, p1 = trace.values[idx - 1], trace.values[idx]
    return float(p0 + (p1 - p0) * (t - t0) / (t1 - t0)), None


def gaps_for(trace: Trace, max_gap_s: float):
    if len(trace.times) < 2:
        return []
    deltas = np.diff(trace.times)
    return [(float(a), float(b)) for a, b, d in zip(trace.times[:-1], trace.times[1:], deltas) if float(d) > max_gap_s]


def gap_overlap(start, end, gaps):
    return [gap for gap in gaps if start < gap[1] and end > gap[0]]


def segments_avoiding_gaps(start, end, gaps):
    boundaries = [start]
    for a, b in gaps:
        if start < a < end:
            boundaries.append(a)
        if start < b < end:
            boundaries.append(b)
    boundaries.append(end)
    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right > left:
            segments.append([float(left), float(right)])
    return segments


def integrate_trace(trace: Trace, start, end, baseline_mode: str = "zero", baseline_value: float = 0.0):
    mask = (trace.times >= start) & (trace.times <= end)
    ts = trace.times[mask]
    ps = trace.values[mask]
    boundary = []
    for t in (start, end):
        value, err = interpolate(trace, float(t))
        if err:
            return None, [err]
        boundary.append(value)
    if len(ts) == 0 or ts[0] != start:
        ts = np.concatenate(([start], ts))
        ps = np.concatenate(([boundary[0]], ps))
    if ts[-1] != end:
        ts = np.concatenate((ts, [end]))
        ps = np.concatenate((ps, [boundary[1]]))
    corrected = ps - (baseline_value if baseline_mode == "constant" else 0.0)
    return float(np.trapz(corrected, ts)), []


def integrate_injection(injection, sample_trace: Trace, blank_trace: Trace | None, rules: dict, injection_index: int):
    start = qdict(injection["time"], "time", f"injections[{injection_index}].time")[0]["canonical_value"]
    volume_q, diags = qdict(injection["volume"], "volume", f"injections[{injection_index}].volume")
    if diags:
        raise SemanticImpossible("invalid injection", diags)
    end = start + float(rules["integration_window_s"])
    max_gap = float(rules["max_sample_gap_s"])
    gaps = gaps_for(sample_trace, max_gap)
    crossed = gap_overlap(start, end, gaps)
    endpoint_errors = []
    for t, label in ((start, "start"), (end, "end")):
        _, err = interpolate(sample_trace, t)
        if err:
            endpoint_errors.append(diagnostic(
                f"injections[{injection_index}].window.{label}",
                f"integration {label} is not bracketed by finite samples or lies outside the power trace",
                {"time_s": t},
                err,
            ))
    discard_first = bool(rules.get("discard_first")) and injection_index == 0
    blank_rule = rules.get("blank_correspondence", "injection_id_exact")
    blank_required = blank_rule == "injection_id_exact" and not discard_first
    missing_blank = blank_required and blank_trace is None
    diagnostics = []
    diagnostics.extend(endpoint_errors)
    if crossed:
        diagnostics.append(diagnostic(
            f"injections[{injection_index}].window",
            "integration window crosses a data gap; reporting valid non-gap segments only",
            {"gaps_s": crossed},
            "window_crosses_gap",
        ))
    if missing_blank:
        diagnostics.append(diagnostic(
            f"injections[{injection_index}].blank",
            "dilution blank has no corresponding injection under exact injection_id correspondence",
            {"required_id": injection["id"], "rule": blank_rule},
            "blank_correspondence_missing",
        ))

    baseline = injection.get("baseline", {"value": [0.0, "uW"], "mode": "constant"})
    baseline_q, bd = qdict(baseline.get("value", [0, "uW"]), "power", f"injections[{injection_index}].baseline")
    diagnostics.extend(bd)
    baseline_value = baseline_q["canonical_value"] if baseline_q else 0.0

    raw_segments = []
    total = 0.0
    calc_errors = []
    target_segments = segments_avoiding_gaps(start, end, crossed) if crossed else [[float(start), float(end)]]
    for si, (a, b) in enumerate(target_segments):
        area, errs = integrate_trace(sample_trace, a, b, baseline.get("mode", "constant"), baseline_value)
        if errs:
            calc_errors.extend(errs)
        else:
            total += area
            raw_segments.append({"start_s": a, "end_s": b, "area_uJ": area})
    blank_area = None
    blank_segments = []
    if blank_trace is not None:
        blank_start = 0.0
        blank_end = end - start
        bg = gaps_for(blank_trace, max_gap)
        if gap_overlap(blank_start, blank_end, bg):
            diagnostics.append(diagnostic(
                f"injections[{injection_index}].blank.window",
                "blank integration window crosses a blank data gap",
                {"gaps_s": gap_overlap(blank_start, blank_end, bg)},
                "blank_window_crosses_gap",
            ))
        blank_area, errs = integrate_trace(blank_trace, blank_start, blank_end, "zero")
        if errs:
            diagnostics.append(diagnostic(f"injections[{injection_index}].blank", "blank endpoint integration failed", errs, "blank_endpoint"))
        else:
            blank_segments.append({"start_s": blank_start, "end_s": blank_end, "area_uJ": blank_area})

    status = "ok"
    reasons = []
    if discard_first:
        status = "warning"
        reasons.append("first_injection_discarded")
    if crossed or endpoint_errors or calc_errors:
        status = "blocked"
        reasons.extend(["data_gap_or_endpoint"] if crossed or endpoint_errors else [])
    if missing_blank:
        status = "blocked"
        reasons.append("blank_correspondence_missing")
    elif status == "ok":
        reasons.append("blank_correspondence_exact")

    return {
        "injection_id": injection["id"],
        "index": injection_index,
        "status": status,
        "reasons": sorted(set(reasons)),
        "window_s": {"start": start, "end": end},
        "injection_volume_uL": volume_q["canonical_value"],
        "baseline_uW": baseline_value,
        "raw_sample_segments": raw_segments,
        "raw_heat_uJ": None if status == "blocked" and (crossed or endpoint_errors) else total,
        "blank_segments": blank_segments,
        "blank_heat_uJ": blank_area,
        "net_heat_uJ": None if (status == "blocked" or blank_area is None) else total - blank_area,
        "blank_rule": blank_rule,
        "diagnostics": diagnostics,
        "rules_version": RULES_VERSION,
    }
