"""Injection integration over the raw power curve.

Hard rules (domain policy, versioned via rule parameters):

* post-injection peaks **may overlap**: each injection is integrated over the
  window ending at the next injection start; spillover is flagged, not refused.
* an integration window **must not cross a data gap**: such a window is
  refused with a located diagnostic; nothing is extrapolated.
* the first injection is discarded when the rule says so (configurable).
* a window ending at the trace end before the signal returns to baseline is
  integrated but marked ``open_endpoint`` and excluded from fitting, with the
  reason exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .units import QuantityError

INTEGRATION_PARAMS_KIND = "integration_params"
RULE_VERSION = "integration/1"


@dataclass(frozen=True)
class IntegrationParams:
    discard_first: bool = True
    baseline_return_tolerance_uW: float = 1.0
    overlap_tail_fraction: float = 0.1
    gap_factor: float = 3.0
    allow_baseline_extrapolation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": INTEGRATION_PARAMS_KIND,
            "rule_version": RULE_VERSION,
            "discard_first": self.discard_first,
            "baseline_return_tolerance_uW": self.baseline_return_tolerance_uW,
            "gap_factor": self.gap_factor,
            "overlap_tail_fraction": self.overlap_tail_fraction,
            "allow_baseline_extrapolation": self.allow_baseline_extrapolation,
        }


def params_from_dict(raw: dict[str, Any]) -> IntegrationParams:
    return IntegrationParams(
        discard_first=bool(raw.get("discard_first", True)),
        baseline_return_tolerance_uW=float(raw.get("baseline_return_tolerance_uW", 1.0)),
        overlap_tail_fraction=float(raw.get("overlap_tail_fraction", 0.1)),
        gap_factor=float(raw.get("gap_factor", 3.0)),
        allow_baseline_extrapolation=bool(raw.get("allow_baseline_extrapolation", False)),
    )


def _find_gaps(
    times: np.ndarray,
    values: np.ndarray,
    gap_factor: float,
) -> list[tuple[float, float, str]]:
    """Return (start, end, reason) intervals considered missing data."""
    gaps: list[tuple[float, float, str]] = []
    if len(times) >= 2:
        dts = np.diff(times)
        med = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.0
        for i, dt in enumerate(dts):
            if med > 0 and dt > gap_factor * med:
                gaps.append((float(times[i]), float(times[i + 1]), "sampling-gap"))
    bad = np.isnan(values)
    if np.any(bad):
        i = 0
        while i < len(bad):
            if bad[i]:
                j = i
                while j + 1 < len(bad) and bad[j + 1]:
                    j += 1
                gaps.append((float(times[i]), float(times[j]), "nan-point"))
                i = j + 1
            else:
                i += 1
    # Merge overlapping intervals.
    gaps.sort()
    merged: list[tuple[float, float, str]] = []
    for g0, g1, why in gaps:
        if merged and g0 <= merged[-1][1]:
            prev0, _, prev_why = merged[-1]
            merged[-1] = (prev0, max(merged[-1][1], g1), f"{prev_why}+{why}")
        else:
            merged.append((g0, g1, why))
    return merged


def _gap_in_window(
    gaps: list[tuple[float, float, str]], start: float, end: float
) -> tuple[float, float, str] | None:
    """A gap with positive overlap strictly attributable to this window."""
    for g0, g1, why in gaps:
        overlap0 = max(g0, start)
        overlap1 = min(g1, end)
        if overlap1 - overlap0 > 1e-12:
            return (g0, g1, why)
    return None


def integrate(
    trace: dict[str, Any],
    schedule: dict[str, Any],
    baseline: dict[str, Any],
    params: IntegrationParams,
    *,
    role: str = "sample",
) -> dict[str, Any]:
    """Run injection integration.

    Material dicts are the *canonical* forms produced by
    :func:`itc.ingest.normalize_material`.  Returns a serializable run.
    """
    diagnostics: list[dict[str, Any]] = []
    try:
        t = np.asarray(trace["times_s"], dtype=float)
        p = np.asarray(trace["power_uW"], dtype=float)
        bt = np.asarray(baseline["times_s"], dtype=float)
        bv = np.asarray(baseline["baseline_uW"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantityError(
            "corrupt-material",
            f"integrate:{role}",
            f"material missing numeric arrays: {exc}",
        ) from exc
    if len(t) != len(p) or len(bt) != len(bv) or len(t) < 2:
        raise QuantityError(
            "corrupt-material", f"integrate:{role}", "series length mismatch or too short"
        )
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    b_order = np.argsort(bt, kind="stable")
    bt, bv = bt[b_order], bv[b_order]

    dt_raw = np.diff(t)
    med_dt = float(np.median(dt_raw[dt_raw > 0])) if np.any(dt_raw > 0) else 1.0
    gaps = _find_gaps(t, p, params.gap_factor)

    injections = sorted(schedule["injections"], key=lambda z: (z["time_s"], z["index"]))
    trace_end = float(t[-1])
    trace_start = float(t[0])
    windows: list[dict[str, Any]] = []

    covered_bt = (float(bt[0]), float(bt[-1]))
    for pos, inj in enumerate(injections):
        start = float(inj["time_s"])
        end = float(injections[pos + 1]["time_s"]) if pos + 1 < len(injections) else trace_end
        window_id = f"heat:{role}:{inj['injection_id']}"
        record: dict[str, Any] = {
            "window_id": window_id,
            "injection_id": inj["injection_id"],
            "index": inj["index"],
            "ordinal": pos,
            "start_s": start,
            "end_s": end,
            "time_basis": trace.get("time_basis", "seconds since acquisition start"),
            "status": None,
            "heat_uJ": None,
            "volume_uL": inj["volume_uL"],
            "flags": [],
            "diagnostics": [],
        }
        if start < trace_start - 1e-9 or start > trace_end + 1e-9:
            record["status"] = "refused"
            record["diagnostics"].append(
                {
                    "code": "injection-outside-trace",
                    "where": f"injection[{inj['index']}].time_s",
                    "message": "injection time is outside the power trace span",
                    "start_s": start,
                    "trace_span_s": [trace_start, trace_end],
                }
            )
            windows.append(record)
            continue
        if params.discard_first and pos == 0:
            record["status"] = "discarded_first"
            record["flags"].append("first_injection")
            windows.append(record)
            continue
        gap = _gap_in_window(gaps, start, min(end, trace_end))
        if gap is not None:
            record["status"] = "refused"
            record["diagnostics"].append(
                {
                    "code": "window-crosses-data-gap",
                    "where": f"injection[{inj['index']}].window_s",
                    "message": "integration window crosses a data gap; integration refused",
                    "gap_interval_s": [gap[0], gap[1]],
                    "gap_reason": gap[2],
                    "window_interval_s": [start, end],
                }
            )
            windows.append(record)
            continue
        if not params.allow_baseline_extrapolation and (
            start < covered_bt[0] - 1e-9 or min(end, trace_end) > covered_bt[1] + 1e-9
        ):
            record["status"] = "semantic_unresolvable"
            record["diagnostics"].append(
                {
                    "code": "baseline-uncovered",
                    "where": f"injection[{inj['index']}].window_s",
                    "message": "no versioned baseline covers this window and extrapolation is disabled",
                    "baseline_span_s": list(covered_bt),
                    "window_interval_s": [start, end],
                }
            )
            windows.append(record)
            continue

        mask = (t >= start - 1e-9) & (t <= end + 1e-9)
        wt, wp = t[mask], p[mask]
        base = np.interp(wt, bt, bv)
        residual = wp - base
        heat = float(np.trapz(residual, wt))
        record["heat_uJ"] = heat

        tail_n = max(2, int(len(wt) * params.overlap_tail_fraction))
        tail = residual[-tail_n:] if len(wt) >= 2 else residual
        tail_above = bool(np.mean(np.abs(tail) > params.baseline_return_tolerance_uW))
        if tail_above and pos + 1 < len(injections):
            record["flags"].append("peak_overlap")
            record["overlap_with_next_injection_id"] = injections[pos + 1]["injection_id"]
        open_end = pos + 1 == len(injections) and (
            len(wt) < 2
            or bool(np.mean(np.abs(tail) > params.baseline_return_tolerance_uW))
        )
        if open_end:
            record["flags"].append("open_endpoint")
            record["status"] = "integrated_open_endpoint"
            record["diagnostics"].append(
                {
                    "code": "open-endpoint",
                    "where": f"injection[{inj['index']}].end_s",
                    "message": "trace ends before power returns to baseline; value kept but excluded from fits",
                    "tail_mean_uW": float(np.mean(tail)),
                    "tolerance_uW": params.baseline_return_tolerance_uW,
                }
            )
        else:
            record["status"] = "integrated"
        windows.append(record)

    run = {
        "kind": "integration_run",
        "rule_version": RULE_VERSION,
        "role": role,
        "params": params.as_dict(),
        "trace_id": trace["material_id"],
        "schedule_id": schedule["material_id"],
        "baseline_id": baseline["material_id"],
        "time_basis": trace.get("time_basis", "seconds since acquisition start"),
        "median_sample_dt_s": med_dt,
        "trace_span_s": [trace_start, trace_end],
        "windows": windows,
        "diagnostics": diagnostics,
        "input_digest": None,
    }
    return run
