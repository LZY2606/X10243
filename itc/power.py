"""Power-sample normalisation, clock bases and data-gap detection.

The raw series keeps every supplied sample, including unreadable or missing
ones (``p is None``).  Normalisation is strictly additive: it never deletes a
point -- bad points are reported with positional diagnostics and excluded only
from numerical windows downstream.
"""
from __future__ import annotations

from typing import Any

from .errors import Diagnostic
from .units import parse_measurement, convert

GAP_FACTOR = 4.0          # dt > 4 * median cadence counts as a gap
MIN_POINTS = 2

POWER_CANON = "W"
TIME_CANON = "s"


def _parse_point(raw: Any, idx: int, power_unit: str, time_unit: str):
    """Return (t_seconds_or_None, p_watts_or_None, original, diagnostics)."""
    diagnostics: list[Diagnostic] = []
    loc = f"/samples/{idx}"
    if raw is None:
        return None, None, None, [Diagnostic("missing_sample",
                                              "sample row is null", loc,
                                              "warning")]
    if isinstance(raw, dict):
        t_raw = raw.get("t", raw.get("time"))
        p_raw = raw.get("p", raw.get("power"))
        original = {"t": t_raw, "p": p_raw}
        if t_raw is None or p_raw is None:
            which = "t" if t_raw is None else "p"
            diagnostics.append(Diagnostic("missing_sample_field",
                                          f"sample field {which!r} missing",
                                          f"{loc}.{which}", "warning", raw))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        t_raw, p_raw = raw[0], raw[1]
        original = [t_raw, p_raw]
    else:
        return None, None, raw, [Diagnostic("bad_sample_shape",
                                            "sample must be [t, p] or "
                                            "{'t':..,'p':..}", loc, "error", raw)]

    t_s: float | None = None
    p_w: float | None = None
    if t_raw is not None:
        if isinstance(t_raw, dict) or not isinstance(t_raw, (int, float)):
            m, ds = parse_measurement(t_raw, f"{loc}.t")
            diagnostics.extend(ds)
            if m is not None:
                v, ds2 = convert(m, TIME_CANON, f"{loc}.t")
                diagnostics.extend(ds2)
                t_s = v
        else:
            if not time_unit:
                diagnostics.append(Diagnostic("missing_unit",
                                              "series has no time unit",
                                              "/time_unit"))
            else:
                m, ds = parse_measurement({"value": t_raw, "unit": time_unit},
                                          f"{loc}.t")
                diagnostics.extend(ds)
                if m is not None:
                    v, ds2 = convert(m, TIME_CANON, f"{loc}.t")
                    diagnostics.extend(ds2)
                    t_s = v
    if p_raw is not None:
        if isinstance(p_raw, dict) or not isinstance(p_raw, (int, float)):
            m, ds = parse_measurement(p_raw, f"{loc}.p")
            diagnostics.extend(ds)
            if m is not None:
                v, ds2 = convert(m, POWER_CANON, f"{loc}.p")
                diagnostics.extend(ds2)
                p_w = v
        else:
            if not power_unit:
                diagnostics.append(Diagnostic("missing_unit",
                                              "series has no power unit",
                                              "/power_unit"))
            else:
                m, ds = parse_measurement({"value": p_raw, "unit": power_unit},
                                          f"{loc}.p")
                diagnostics.extend(ds)
                if m is not None:
                    v, ds2 = convert(m, POWER_CANON, f"{loc}.p")
                    diagnostics.extend(ds2)
                    p_w = v
    return t_s, p_w, original, diagnostics


def normalize_series(payload: dict[str, Any],
                     prefix: str = "") -> dict[str, Any]:
    """Normalise one raw power series payload.

    Returns a dict with ``t``/``p`` canonical arrays (seconds/watts),
    ``points`` preserving originals, declared/detected ``gaps`` and
    diagnostics.  Missing samples produce gaps and ``None`` coordinates rather
    than being removed.
    """
    diagnostics: list[Diagnostic] = []
    power_unit = payload.get("power_unit") or payload.get("unit") or ""
    time_unit = payload.get("time_unit") or "s"
    raw_samples = payload.get("samples")
    time_basis = payload.get("time_basis", "experiment_clock:t=0 at acquisition start")
    if raw_samples is None:
        diagnostics.append(Diagnostic("missing_samples",
                                      "power series has no samples",
                                      f"{prefix}/samples"))
        raw_samples = []

    points: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_samples):
        t_s, p_w, original, ds = _parse_point(raw, idx, power_unit, time_unit)
        diagnostics.extend(ds)
        points.append({"index": idx, "t_s": t_s, "p_w": p_w,
                       "original": original})

    # ordering checks
    last_t = None
    for pt in points:
        if pt["t_s"] is not None:
            if last_t is not None and pt["t_s"] < last_t:
                diagnostics.append(Diagnostic(
                    "non_monotonic_time",
                    f"sample {pt['index']} time {pt['t_s']} precedes {last_t}",
                    f"{prefix}/samples/{pt['index']}/t", "error"))
            last_t = pt["t_s"]

    gaps = detect_gaps(points, payload.get("declared_gaps", []), prefix)
    for g in gaps:
        if g["kind"] in ("missing_sample", "declared"):
            diagnostics.append(Diagnostic(
                "data_gap",
                f"data gap between samples {g['after_index']} and "
                f"{g['before_index']} ({g['duration_s']:.3g} s)",
                g["location"], "warning"))

    if len(points) < MIN_POINTS:
        diagnostics.append(Diagnostic("series_too_short",
                                      f"need >= {MIN_POINTS} samples",
                                      f"{prefix}/samples", "error"))

    return {
        "series_id": payload.get("series_id", "power"),
        "power_unit_original": power_unit,
        "time_unit_original": time_unit,
        "time_basis": time_basis,
        "clock_zero": payload.get("clock_zero", "t=0 acquisition start"),
        "points": points,
        "t": [p["t_s"] for p in points],
        "p": [p["p_w"] for p in points],
        "gaps": gaps,
        "diagnostics": diagnostics,
    }


def detect_gaps(points: list[dict[str, Any]],
                declared_gaps: list[dict[str, Any]] | None = None,
                prefix: str = "") -> list[dict[str, Any]]:
    """Find missing samples, declared gaps and cadence gaps.

    A gap is the open interval (t_after .. t_before) with no usable samples.
    Integration windows must never span a gap.
    """
    gaps: list[dict[str, Any]] = []
    usable = [p for p in points if p["t_s"] is not None]
    dts = [usable[i + 1]["t_s"] - usable[i]["t_s"]
           for i in range(len(usable) - 1)
           if usable[i]["p_w"] is not None and usable[i + 1]["p_w"] is not None]
    dts = sorted(d for d in dts if d > 0)
    median_dt = dts[len(dts) // 2] if dts else None

    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a["t_s"] is None or b["t_s"] is None:
            gaps.append({"after_index": a["index"], "before_index": b["index"],
                         "t_after": a["t_s"], "t_before": b["t_s"],
                         "duration_s": None, "kind": "missing_sample",
                         "location": f"{prefix}/samples/{b['index']}"})
            continue
        duration = b["t_s"] - a["t_s"]
        declared = any(
            g.get("after_index") == a["index"] for g in (declared_gaps or [])
        )
        if declared:
            gaps.append({"after_index": a["index"], "before_index": b["index"],
                         "t_after": a["t_s"], "t_before": b["t_s"],
                         "duration_s": duration, "kind": "declared",
                         "location": f"{prefix}/samples/{b['index']}"})
        elif median_dt and duration > GAP_FACTOR * median_dt:
            gaps.append({"after_index": a["index"], "before_index": b["index"],
                         "t_after": a["t_s"], "t_before": b["t_s"],
                         "duration_s": duration, "kind": "cadence",
                         "location": f"{prefix}/samples/{b['index']}"})
    return gaps


def baseline_value(baseline_norm: dict[str, Any], t_s: float) -> float:
    """Piecewise-linear / constant baseline evaluation at time t (watts)."""
    segments = baseline_norm.get("segments")
    if not segments:
        return baseline_norm.get("constant_w", 0.0)
    active = segments[0]
    for seg in segments:
        if seg["t0_s"] <= t_s:
            active = seg
    t0 = active["t0_s"]
    return active["intercept_w"] + active["slope_w_per_s"] * (t_s - t0)


def normalize_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    kind = payload.get("kind", "constant")
    out: dict[str, Any] = {
        "baseline_id": payload.get("baseline_id", "baseline"),
        "kind": kind,
        "time_basis": payload.get("time_basis", "experiment_clock"),
        "diagnostics": diagnostics,
        "source": payload,
    }
    if kind == "constant":
        m, ds = parse_measurement(payload.get("value"), "/baseline/value")
        diagnostics.extend(ds)
        const = None
        if m is not None:
            v, ds2 = convert(m, POWER_CANON, "/baseline/value")
            diagnostics.extend(ds2)
            const = v
        out["constant_w"] = const if const is not None else 0.0
        out["measurement"] = m.to_dict() if m else None
        if const is None:
            diagnostics.append(Diagnostic("missing_baseline",
                                          "constant baseline value unusable",
                                          "/baseline/value"))
    elif kind in ("segmented", "linear"):
        segments = []
        for i, seg in enumerate(payload.get("segments", [])):
            loc = f"/baseline/segments/{i}"
            entry = {"index": i}
            tm, d1 = parse_measurement(seg.get("t0"), f"{loc}/t0")
            diagnostics.extend(d1)
            im, d2 = parse_measurement(seg.get("intercept"), f"{loc}/intercept")
            diagnostics.extend(d2)
            entry["t0_s"], _ = (convert(tm, TIME_CANON, f"{loc}/t0")
                                if tm else (None, d1))
            iw, d3 = (convert(im, POWER_CANON, f"{loc}/intercept")
                      if im else (None, d2))
            diagnostics.extend(d3)
            entry["intercept_w"] = iw if iw is not None else 0.0
            slope_raw = seg.get("slope", {"value": 0.0, "unit": "W/s"})
            sm, d4 = parse_measurement(slope_raw, f"{loc}/slope")
            diagnostics.extend(d4)
            entry["slope_w_per_s"] = 0.0
            if sm is not None:
                slope_v, d5 = convert(sm, "W/s", f"{loc}/slope")
                diagnostics.extend(d5)
                if slope_v is not None:
                    entry["slope_w_per_s"] = slope_v
            segments.append(entry)
        segments.sort(key=lambda s: (s["t0_s"] is None, s["t0_s"]))
        out["segments"] = segments
    else:
        diagnostics.append(Diagnostic("unknown_baseline_kind",
                                      f"unknown baseline kind {kind!r}",
                                      "/baseline/kind"))
    return out
