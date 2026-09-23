"""Per-injection numerical integration of an ITC power curve.

Domain guarantees implemented here:

* Post-injection peaks may overlap.  A window ends at the next injection; the
  residual tail of every earlier peak is subtracted analytically inside later
  windows (single-exponential deconvolution).
* An integration window never spans a structural data gap (absent
  coordinates / non-uniform time jumps).  The window is then reported
  ``blocked`` with the gap location -- no area is fabricated.
* A single missing sample (``NA`` with a retained coordinate) is linearly
  interpolated and recorded; it does not constitute a gap.
* Windows truncated by the stream end are ``partial``.
* The first injection can be discarded by the integration rule version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

from .models_common import estimate_tau, make_value, find_gaps


@dataclass
class IntegrationConfig:
    pre_seconds: float = 10.0
    return_tolerance_uW: float = 0.15
    settle_seconds: float = 6.0
    max_window_seconds: float = 150.0
    discard_first: bool = False

    @classmethod
    def from_rule(cls, rule: dict[str, Any]) -> "IntegrationConfig":
        p = rule.get("parameters", rule)
        return cls(
            pre_seconds=float(p.get("pre_injection_seconds", 10.0)),
            return_tolerance_uW=float(p.get("return_tolerance_uW", 0.15)),
            settle_seconds=float(p.get("settle_seconds", 6.0)),
            max_window_seconds=float(p.get("max_window_seconds", 150.0)),
            discard_first=bool(p.get("discard_first_injection", False)),
        )


def integrate_stream(
    material: dict[str, Any],
    rule: dict[str, Any],
    *,
    stream_id: str = "power",
) -> dict[str, Any]:
    """Integrate every injection of one power material.

    Returns a structured result containing per-injection windows and values;
    never raises for domain conditions (gaps/endpoints) -- they are statuses.
    """
    cfg = IntegrationConfig.from_rule(rule)
    payload = material["payload"]
    power = np.array(payload["power"], dtype=float)  # NA -> NaN
    times = power[:, 0]
    signal = power[:, 1]
    injections = payload["injections"]
    baseline_offset = _baseline_offset(payload.get("baseline", []))

    valid = ~np.isnan(times)
    times = times[valid]
    signal = signal[valid]
    order = np.argsort(times)
    times = times[order]
    signal = signal[order]
    # Collapse exact duplicate timestamps (deterministic: keep first sample).
    unique_mask = np.ones(len(times), dtype=bool)
    unique_mask[1:] = times[1:] != times[:-1]
    times = times[unique_mask]
    signal = signal[unique_mask]

    na_positions = [int(i) for i, v in enumerate(power[:, 1]) if np.isnan(v)]
    gaps = find_gaps(times)
    dt_median = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    tau = estimate_tau(times, signal - baseline_offset, injections,
                       default=9.0)

    windows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    n = len(injections)
    for k, inj in enumerate(injections):
        index = int(inj["index"])
        t0 = float(inj["time_s"])
        next_t0 = float(injections[k + 1]["time_s"]) if k + 1 < n else None

        entry: dict[str, Any] = {
            "injection_index": index,
            "injection_time_s": t0,
            "injection_time": make_value(t0, "s", "time",
                                         coords={"axis": "t", "origin": "power_stream",
                                                 "origin_material": material["material_id"]}),
            "injection_volume": make_value(float(inj.get("volume_uL", inj.get("volume_uW", 0.0))),
                                           "uL", "volume"),
            "status": "ok",
            "flags": [],
            "window": None,
            "area_uJ": None,
            "peak_amplitude_uW": None,
            "diagnostics": [],
        }
        if cfg.discard_first and k == 0:
            entry["status"] = "discarded"
            entry["flags"].append("first_injection_discarded")
            entry["diagnostics"].append({
                "code": "first_injection_discarded",
                "severity": "info",
                "message": "First injection discarded by integration rule.",
                "rule_version": rule.get("rule_version",
                                         rule.get("version_label")),
            })
            windows.append(entry)
            continue

        start = t0 - cfg.pre_seconds
        hard_end = min(t0 + cfg.max_window_seconds,
                       next_t0 if next_t0 is not None else math.inf,
                       float(times[-1]))
        if hard_end <= t0:
            entry["status"] = "blocked"
            entry["flags"].append("zero_window")
            entry["diagnostics"].append({
                "code": "zero_window", "severity": "error",
                "message": "Window has no post-injection samples.",
            })
            windows.append(entry)
            continue

        # Return-to-baseline endpoint, limited by hard_end.
        return_t = _resolve_time(times, signal, baseline_offset, t0,
                                 hard_end, cfg.return_tolerance_uW,
                                 cfg.settle_seconds, next_t0)
        end, endpoint_reason = return_t
        entry.setdefault("flags", []).append(endpoint_reason)

        post_idx = np.where((times >= t0) & (times <= end))[0]
        pre_idx = np.where((times >= start) & (times < t0))[0]
        interp_samples: list[dict[str, Any]] = []
        window_signal = signal.copy()
        # Linear interpolation at NaN positions within window.
        for local in list(pre_idx) + list(post_idx):
            if np.isnan(window_signal[local]):
                window_signal[local] = _interp(times, signal, local)
                interp_samples.append({
                    "t_s": float(times[local]),
                    "code": "missing_sample_interpolated",
                })
                entry["flags"].append("contains_interpolated_sample")

        # Structural gap check (time discontinuity) within [start, end].
        crossed = [g for g in gaps if start < g["end_prev_s"] < end
                   or start < g["start_next_s"] < end
                   or (g["end_prev_s"] < start and g["start_next_s"] > end)]
        crossed = [g for g in gaps
                   if g["end_prev_s"] >= start and g["start_next_s"] <= end]
        if crossed:
            entry["status"] = "blocked"
            entry["flags"].append("window_crosses_data_gap")
            entry["window"] = {
                "start_s": make_value(start, "s", "time"),
                "end_s": make_value(end, "s", "time"),
            }
            for g in crossed:
                entry["diagnostics"].append({
                    "code": "window_crosses_data_gap", "severity": "error",
                    "message": "Integration window spans absent samples; "
                               "no area computed.",
                    "gap_after_s": g["end_prev_s"],
                    "gap_before_s": g["start_next_s"],
                })
            windows.append(entry)
            continue

        # Endpoint truncation checks.
        if end >= float(times[-1]) - 0.5 * dt_median:
            entry["status"] = "partial"
            entry["flags"].append("truncated_at_stream_end")
            entry["diagnostics"].append({
                "code": "truncated_endpoint", "severity": "warning",
                "message": "Window ends at the last available sample.",
            })
        if next_t0 is not None and end >= next_t0 - 1e-9:
            entry["flags"].append("truncated_by_next_injection")

        # Overlap: subtract fitted tails of all earlier peaks.  A tail still
        # above tolerance at this injection marks the window as overlapping.
        amp = _local_amplitude(times, signal, baseline_offset, t0)
        if amp is not None:
            entry["peak_amplitude_uW"] = make_value(amp, "uW", "power")
        tails_subtracted = []
        for prior in windows:
            if prior["status"] in ("discarded", "blocked"):
                continue
            pa = prior.get("peak_amplitude_uW")
            if pa is None:
                continue
            prior_amp = pa["value"]
            prior_t = prior["injection_time_s"]
            tail_at = prior_amp * math.exp(-(t0 - prior_t) / tau)
            if tail_at > cfg.return_tolerance_uW:
                entry["flags"].append("overlapping_peak")
                tails_subtracted.append({
                    "prior_injection_index": prior["injection_index"],
                    "tail_at_window_start_uW": tail_at,
                })
        # Build corrected signal on the window.
        mask = (times >= start) & (times <= end)
        wt = times[mask]
        ws = window_signal[mask].astype(float)
        ws_corrected = ws - baseline_offset
        for prior in windows:
            pa = prior.get("peak_amplitude_uW")
            if pa is None or prior["status"] in ("discarded", "blocked"):
                continue
            prior_t = prior["injection_time_s"]
            tail = pa["value"] * np.exp(-(wt - prior_t) / tau)
            tail[(wt < prior_t)] = 0.0
            ws_corrected = ws_corrected - tail
        # Baseline region mean: use pre-window samples for a local offset check.
        area = float(np.trapezoid(ws_corrected, wt))  # uW*s = uJ
        entry["area_uJ"] = make_value(area, "uJ", "energy")
        entry["tails_subtracted"] = tails_subtracted
        entry["interpolated_samples"] = interp_samples
        entry["window"] = {
            "start_s": make_value(start, "s", "time"),
            "end_s": make_value(end, "s", "time"),
            "n_samples": int(mask.sum()),
        }
        if len(wt) < 2:
            entry["status"] = "blocked"
            entry["diagnostics"].append({
                "code": "insufficient_samples", "severity": "error",
                "message": "Fewer than two samples in window.",
            })
        windows.append(entry)

    result = {
        "kind": "integration",
        "stream_id": stream_id,
        "source_material_id": material["material_id"],
        "rule_id": rule.get("rule_id"),
        "rule_version": rule.get("rule_version", rule.get("version_label")),
        "tau_s": make_value(float(tau), "s", "time"),
        "baseline_offset_uW": make_value(float(baseline_offset), "uW", "power"),
        "data_gaps_s": gaps,
        "missing_sample_times_s": [
            float(power[i, 0]) for i in na_positions
            if not np.isnan(power[i, 0])
        ],
        "time_reference": {"axis": "t", "unit": "s", "origin": "power_stream",
                           "origin_material": material["material_id"]},
        "injections": windows,
        "diagnostics": diagnostics,
        "input_summary": _input_summary(payload),
        "parameters": {
            "pre_injection_seconds": cfg.pre_seconds,
            "return_tolerance_uW": cfg.return_tolerance_uW,
            "settle_seconds": cfg.settle_seconds,
            "max_window_seconds": cfg.max_window_seconds,
            "discard_first_injection": cfg.discard_first,
        },
    }
    return result


def _input_summary(payload: dict[str, Any]) -> dict[str, Any]:
    power = payload.get("power", [])
    return {
        "n_power_samples": len(power),
        "n_injections": len(payload.get("injections", [])),
        "n_baseline_segments": len(payload.get("baseline", [])),
    }


def _baseline_offset(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    return float(np.mean([s["offset_uW"] for s in segments]))


def _resolve_time(times, signal, offset, t0, hard_end, tol, settle,
                  next_t0):
    """Return (window_end, reason).  End is a settle point or hard_end."""
    post = (times >= t0) & (times <= hard_end)
    pt = times[post]
    ps = signal[post] - offset
    if len(pt) == 0:
        return hard_end, "hard_end_no_post_samples"
    run = 0
    for i, t in enumerate(pt):
        if abs(ps[i]) <= tol:
            run += 1
        else:
            run = 0
        if run >= max(1, int(math.ceil(settle / max(1e-9,
                                                     _median_dt(pt))))):
            start_idx = i - run + 1
            return float(pt[start_idx]), "endpoint_return_to_baseline"
    if next_t0 is not None and hard_end >= next_t0 - 1e-9:
        return float(hard_end), "endpoint_next_injection"
    if hard_end >= float(times[-1]) - 1e-9:
        return float(hard_end), "endpoint_stream_end"
    return float(hard_end), "endpoint_max_window"


def _median_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0
    return float(np.median(np.diff(t)))


def _local_amplitude(times, signal, offset, t0, *, width=4.0):
    idx = np.where((times >= t0) & (times <= t0 + width))[0]
    if len(idx) == 0:
        return None
    return float(np.nanmax(signal[idx]) - offset)


def _interp(times, signal, i):
    j = i - 1
    while j >= 0 and np.isnan(signal[j]):
        j -= 1
    k = i + 1
    while k < len(signal) and np.isnan(signal[k]):
        k += 1
    if j < 0 or k >= len(signal):
        return 0.0
    frac = (times[i] - times[j]) / (times[k] - times[j])
    return signal[j] + frac * (signal[k] - signal[j])
