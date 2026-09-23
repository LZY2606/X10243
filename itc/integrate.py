"""Per-injection integration of the power curve.

Domain rules enforced here (also encoded as machine-readable events upstream):

* **Post-injection peaks may overlap.**  A peak is integrated in its own
  window; when the next injection arrives before baseline return, the window is
  clipped at the next injection time and ``clipped_by_next_injection`` is set.
* **An integration window never spans a data gap.**  Each window is intersected
  with the valid (gap-free) sample intervals; if the interior is broken, only
  the interval containing the injection is integrated and the missing tail is
  reported.  Too few surviving points invalidate the integral.
* Integration endpoints (start/end time, baseline-return point, clip reasons)
  are explicit outputs -- never implicit.

Numerics: trapezoidal integration of baseline-subtracted power in joules,
reported canonically in µcal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import Diagnostic
from .units import convert, parse_measurement
from .power import baseline_value

JOULE_PER_UCAL = 4.184e-6


def _valid_intervals(t: np.ndarray, p: np.ndarray,
                     gaps: list[dict[str, Any]]) -> list[tuple[float, float, int, int]]:
    """Maximal (t_lo, t_hi, i_lo, i_hi) intervals without a gap inside."""
    good = np.array([x is not None for x in t], dtype=bool) & np.array(
        [x is not None for x in p], dtype=bool)
    idx = np.where(good)[0]
    if idx.size == 0:
        return []
    cuts = {(g["after_index"], g["before_index"]) for g in gaps}
    intervals = []
    start = idx[0]
    prev = idx[0]
    for cur in idx[1:]:
        if (prev, cur) in cuts:
            intervals.append((float(t[start]), float(t[prev]), int(start), int(prev)))
            start = cur
        prev = cur
    intervals.append((float(t[start]), float(t[prev]), int(start), int(prev)))
    return intervals


def _estimate_noise(p: np.ndarray) -> float:
    if p.size < 4:
        return 0.0
    diffs = np.diff(p)
    return 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) or 1e-12


def _baseline_return(t: np.ndarray, y: np.ndarray, start: int, hard_end: float,
                     start_delay: float, noise: float,
                     band_sigma: float = 4.0,
                     dwell: int = 6) -> tuple[float | None, int | None, str]:
    """First time after (start+delay) where signal stays within noise band."""
    threshold = band_sigma * noise
    t0 = t[start] + start_delay
    i = start
    while i < len(t) and t[i] < t0:
        i += 1
    if i >= len(t):
        return None, None, "no_samples_after_start"
    run = 0
    for j in range(i, len(t)):
        if t[j] > hard_end:
            break
        if abs(y[j]) <= threshold:
            run += 1
            if run >= dwell:
                k = j - dwell + 1
                return float(t[k]), int(k), "baseline_return"
        else:
            run = 0
    return None, None, "no_return_before_limit"


def integrate_injections(power_norm: dict[str, Any],
                         schedule_norm: dict[str, Any],
                         baseline_norm: dict[str, Any],
                         rules: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
    """Integrate every injection. Deterministic for identical inputs."""
    rules = rules or {}
    start_delay_s = float(rules.get("start_delay_s", 0.0))
    band_sigma = float(rules.get("baseline_band_sigma", 4.0))
    dwell = int(rules.get("baseline_dwell_points", 6))
    diagnostics: list[Diagnostic] = []

    t_raw = list(power_norm["t"])
    p_raw = list(power_norm["p"])
    gaps = power_norm.get("gaps", [])
    t = np.array([x if x is not None else np.nan for x in t_raw], dtype=float)
    p = np.array([x if x is not None else np.nan for x in p_raw], dtype=float)
    if t.size == 0:
        diagnostics.append(Diagnostic("no_power_samples",
                                      "cannot integrate: no power samples",
                                      "/power/samples"))
        return {"integrations": [], "diagnostics": diagnostics}

    base = np.array([baseline_value(baseline_norm, float(ti))
                     if not math.isnan(ti) else np.nan for ti in t])
    y = p - base
    noise = _estimate_noise(p[~np.isnan(p)])
    intervals = _valid_intervals(t_raw, p_raw, gaps)
    injections = schedule_norm["injections"]

    results = []
    for pos, inj in enumerate(injections):):
        t_inj = inj["time_s"]
        label = inj["label"]
        loc = f"/injections/{inj['index']}"
        entry: dict[str, Any] = {
            "ordinal": inj["ordinal"],
            "label": label,
            "schedule_index": inj["index"],
            "injection_time_s": t_inj,
            "volume_l": inj["volume_l"],
            "diagnostics": [],
            "valid": False,
        }
        if t_inj is None:
            entry["diagnostics"].append(Diagnostic(
                "injection_time_missing",
                f"injection {label}: time unusable, not integrated",
                f"{loc}/time"))
            results.append(entry)
            continue

        # hard window: [t_inj + start_delay .. next injection start)
        hard_end = (injections[pos + 1]["time_s"]
                    if pos + 1 < len(injections)
                    and injections[pos + 1]["time_s"] is not None
                    else float(np.nanmax(t)))
        start_time = t_inj + start_delay_s

        # locate starting sample
        start_idx = int(np.searchsorted(t, start_time, side="left"))
        if start_idx >= len(t) or math.isnan(t[start_idx]):
            # try to find any valid point at/after start before hard_end
            candidates = [k for k in range(start_idx, len(t))
                          if not math.isnan(t[k]) and t[k] <= hard_end]
            if not candidates:
                entry["diagnostics"].append(Diagnostic(
                    "no_samples_in_window",
                    f"injection {label}: window contains no usable samples",
                    f"{loc}"))
                entry.update({"window_start_s": start_time,
                              "window_end_s": hard_end})
                results.append(entry)
                continue
            start_idx = candidates[0]

        ret_t, ret_i, ret_reason = _baseline_return(
            t, y, start_idx, hard_end, start_delay_s, noise,
            band_sigma, dwell)
        end_time = ret_t if ret_t is not None else hard_end
        endpoint_reason = ret_reason
        clipped_next = False
        if ret_t is None and pos + 1 < len(injections):
            clipped_next = True
            endpoint_reason = "clipped_at_next_injection"
        elif ret_t is None:
            endpoint_reason = "clipped_at_series_end"

        # clip window into the valid (gap-free) interval containing start
        chosen = None
        for (lo, hi, ilo, ihi) in intervals:
            if lo <= t[start_idx] <= hi or (ilo == start_idx):
                if lo <= start_time <= hi or hi >= t[start_idx]:
                    chosen = (lo, hi, ilo, ihi)
                    break
        gap_truncated = False
        if chosen is None:
            entry["diagnostics"].append(Diagnostic(
                "window_inside_gap",
                f"injection {label}: window start falls in a data gap",
                f"{loc}"))
            entry.update({"window_start_s": start_time,
                          "window_end_s": end_time,
                          "endpoint_reason": endpoint_reason})
            results.append(entry)
            continue
        lo, hi, ilo, ihi = chosen
        wlo_t, whi_t = start_time, min(end_time, hi)
        if whi_t < end_time - 1e-12:
            gap_truncated = True
            endpoint_reason = "truncated_at_data_gap"
            entry["diagnostics"].append(Diagnostic(
                "integration_crosses_gap",
                f"injection {label}: peak tail crosses a data gap; "
                "window truncated, integral is a lower bound",
                f"{loc}", "warning"))

        mask = (t >= wlo_t - 1e-12) & (t <= whi_t + 1e-12) & \
               np.array([lo <= tk <= hi for tk in t])
        wt = t[mask]
        wy = y[mask]
        wp = p[mask]
        if wt.size < 2:
            entry["diagnostics"].append(Diagnostic(
                "insufficient_points",
                f"injection {label}: < 2 points survive window/gap clipping",
                f"{loc}"))
            entry.update({"window_start_s": wlo_t, "window_end_s": whi_t,
                          "endpoint_reason": endpoint_reason,
                          "n_points": int(wt.size)})
            results.append(entry)
            continue

        q_j = float(np.trapezoid(wy, wt)) if hasattr(np, "trapezoid") \
            else float(np.trapz(wy, wt))
        q_raw_j = float(np.trapezoid(wp, wt)) if hasattr(np, "trapezoid") \
            else float(np.trapz(wp, wt))
        entry.update({
            "valid": True,
            "window_start_s": float(wt[0]),
            "window_end_s": float(wt[-1]),
            "requested_start_s": start_time,
            "requested_end_s": float(end_time),
            "endpoint_reason": endpoint_reason,
            "baseline_return_s": ret_t,
            "clipped_by_next_injection": clipped_next,
            "gap_truncated": gap_truncated,
            "overlaps_next_peak": bool(clipped_next),
            "n_points": int(wt.size),
            "sample_index_start": int(np.where(mask)[0][0]),
            "sample_index_end": int(np.where(mask)[0][-1]),
            "q_j": q_j,
            "q_ucal": q_j / JOULE_PER_UCAL,
            "q_raw_j": q_raw_j,
            "q_raw_ucal": q_raw_j / JOULE_PER_UCAL,
            "noise_w": noise,
            "baseline_band_w": band_sigma * noise,
        })
        results.append(entry)
    return {"integrations": results,
            "diagnostics": [d for r in results for d in r["diagnostics"]]}


def normalize_rules(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Integration rules version (start delay, baseline band ...)."""
    payload = payload or {}
    return {
        "rules_id": payload.get("rules_id", "integration_rules"),
        "start_delay_s": float(payload.get("start_delay_s", 0.0)),
        "baseline_band_sigma": float(payload.get("baseline_band_sigma", 4.0)),
        "baseline_dwell_points": int(payload.get("baseline_dwell_points", 6)),
        "blank_matching": payload.get("blank_matching", "ordinal_and_volume"),
        "source": payload,
    }
