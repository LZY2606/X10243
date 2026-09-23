"""Per-injection integration of the power curve.

Hard rules exercised by the acceptance suite:

* peaks from successive injections **may overlap**; integration windows are
  clipped so a later injection always ends where the next injection starts;
* an integration window must **never cross a data gap** - instead the result
  is ``unresolved`` with a located diagnostic (no fake value);
* the first injection can be **discarded** by an explicit decision; the raw
  integration is retained but not fed to blank subtraction/fitting;
* integration **endpoints** are detected (return to baseline) and can be
  manually overridden; overrides beyond coverage / inside a gap are rejected.

Inputs keep their imported units; this is the derived (normalised) layer and
works in seconds and microcal/s, returning heats in cal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import stable_float
from .errors import BudgetExhausted, SemanticImpasse, diagnostic
from .rules import RuleSet


def _median_dt(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def find_gaps(times: np.ndarray, rules: RuleSet) -> list[tuple[float, float]]:
    """Return ``[(t_after_gap, t_before_gap), ...]`` sorted by time."""
    dt = np.diff(times)
    med = float(np.median(dt)) if dt.size else 0.0
    if med <= 0:
        return []
    idx = np.where(dt > rules.gap_factor * med)[0]
    return [(float(times[i + 1]), float(times[i])) for i in idx]


def _gap_at(gaps: list[tuple[float, float]], t0: float, t1: float) -> tuple[
        float, float] | None:
    for after, before in gaps:
        if t0 < before and after < t1:  # gap interior strictly inside window
            return (after, before)
    return None


def _estimate_baseline(powers: np.ndarray, end: int) -> tuple[float, float]:
    """Pre-injection baseline level and noise (last samples before the shot)."""
    lo = max(0, end - 30)
    segment = powers[lo:end] if end > lo else powers[: max(1, end)]
    if segment.size == 0:
        segment = powers[: min(8, powers.size)]
    return float(np.median(segment)), float(np.std(segment)) + 1e-12


def _peak_end(times: np.ndarray, powers: np.ndarray, start: int, hard_end: int,
              base: float, sigma: float, rules: RuleSet) -> tuple[int | None, str]:
    """Find first sustained return to baseline, bounded by next injection."""
    tol = rules.return_tol * sigma
    below_since: int | None = None
    need = max(2, int(rules.min_window_samples))
    j = start
    n = min(hard_end, times.size)
    while j < n:
        if abs(powers[j] - base) <= tol:
            if below_since is None:
                below_since = j
            elif j - below_since + 1 >= need:
                return below_since, "returned_to_baseline"
        else:
            below_since = None
        j += 1
    return None, "next_injection_or_end"


def integrate_injection(times: np.ndarray, powers: np.ndarray,
                        shot_time: float, next_shot_time: float | None,
                        rules: RuleSet, *, gaps: list[tuple[float, float]] | None = None,
                        end_override: float | None = None,
                        series_t0: float = 0.0) -> dict[str, Any]:
    """Integrate one injection. Always returns a structured result dict.

    Status values:
      * ``ok``                 - usable integrated heat
      * ``clipped_overlap``    - peaks overlapped; window ended at next shot
      * ``unresolved_gap``     - window would cross a data gap
      * ``out_of_coverage``    - window lies outside sampled time
      * ``insufficient``       - too few points / no peak
    """
    t = np.asarray(times, dtype=float)
    p = np.asarray(powers, dtype=float)
    gaps = gaps if gaps is not None else find_gaps(t, rules)
    diags: list[dict[str, Any]] = []

    start = int(np.searchsorted(t, shot_time, side="left"))
    if start >= t.size:
        return _bad("out_of_coverage", shot_time, next_shot_time,
                    [diagnostic("window_outside_data",
                                "injection time is after the last sample",
                                injection_time=shot_time)])

    next_idx = (int(np.searchsorted(t, next_shot_time, side="left"))
                if next_shot_time is not None else t.size)
    hard_end = min(next_idx, t.size)

    base, sigma = _estimate_baseline(p, start)

    # Data gap between injection and any legal endpoint?
    hard_t1 = float(t[hard_end - 1]) if hard_end > start else shot_time
    gap = _gap_at(gaps, shot_time, hard_t1)
    if gap is not None:
        after, before = gap
        return _bad("unresolved_gap", shot_time, next_shot_time,
                    [diagnostic("window_crosses_gap",
                                "integration window would cross a data gap; "
                                "endpoint must be set before the gap",
                                injection_time=shot_time, gap_start=before,
                                gap_end=after)],
                    baseline=base, sigma=sigma, gap={"start": before, "end": after})

    end_idx, reason = _peak_end(t, p, start, hard_end, base, sigma, rules)
    endpoint_mode = reason
    if end_idx is None:
        # overlap clipping: the peak never returned before the next injection
        if next_shot_time is not None and next_idx < t.size:
            end_idx = hard_end
            endpoint_mode = "next_injection_clip"
            diags.append(diagnostic("overlapping_peaks",
                                    "peak did not return to baseline before the "
                                    "next injection; window clipped there",
                                    injection_time=shot_time,
                                    next_injection_time=next_shot_time))
        else:
            return _bad("insufficient", shot_time, next_shot_time,
                        [diagnostic("no_return_to_baseline",
                                    "no return to baseline within coverage",
                                    injection_time=shot_time)],
                        baseline=base, sigma=sigma)

    if end_override is not None:
        ov = float(end_override)
        ov_idx = int(np.searchsorted(t, ov, side="left"))
        if ov <= shot_time:
            raise SemanticImpasse(
                "endpoint override must be after the injection time",
                locations=[{"injection_time": shot_time, "endpoint": ov}])
        if ov_idx >= t.size:
            raise SemanticImpasse(
                "endpoint override is beyond sampled data",
                locations=[{"endpoint": ov, "last_sample": float(t[-1])}])
        ov_gap = _gap_at(gaps, shot_time, ov)
        if ov_gap is not None:
            raise SemanticImpasse(
                "endpoint override crosses a data gap",
                locations=[{"endpoint": ov, "gap_start": ov_gap[1],
                            "gap_end": ov_gap[0]}])
        if next_shot_time is not None and ov > next_shot_time + 1e-9:
            raise SemanticImpasse(
                "endpoint override crosses into the next injection",
                locations=[{"endpoint": ov,
                            "next_injection_time": next_shot_time}])
        end_idx = ov_idx
        endpoint_mode = "manual_override"

    sl = slice(start, end_idx + 1)
    wt, wp = t[sl], p[sl]
    if wt.size < rules.min_window_samples:
        return _bad("insufficient", shot_time, next_shot_time,
                    [diagnostic("too_few_samples",
                                f"fewer than {rules.min_window_samples} samples "
                                "in the integration window",
                                injection_time=shot_time, n_samples=int(wt.size))],
                    baseline=base, sigma=sigma)

    area = float(np.trapezoid(wp - base, wt))  # µcal (power in µcal/s * s)
    heat_cal = area * 1e-6
    # uncertainty from baseline noise propagated over the window width
    width = float(wt[-1] - wt[0]) if wt.size > 1 else 0.0
    uncertainty = abs(sigma) * max(width, 0.0) * 1e-6
    status = "clipped_overlap" if endpoint_mode == "next_injection_clip" else "ok"
    return {
        "status": status,
        "shot_time": stable_float(shot_time - series_t0),
        "next_shot_time": (stable_float(next_shot_time - series_t0)
                           if next_shot_time is not None else None),
        "window_start_index": start,
        "window_end_index": int(end_idx),
        "window_start_time": stable_float(float(wt[0]) - series_t0),
        "window_end_time": stable_float(float(wt[-1]) - series_t0),
        "endpoint_mode": endpoint_mode,
        "baseline_level": stable_float(base),
        "baseline_sigma": stable_float(sigma),
        "n_samples": int(wt.size),
        "heat_cal": stable_float(heat_cal),
        "heat_uncertainty_cal": stable_float(uncertainty),
        "heat_unit": "cal",
        "time_basis": "series_first_sample",
        "overlaps_next": status == "clipped_overlap",
        "diagnostics": diags,
    }


def _bad(status: str, shot: float, nxt: float | None,
         diags: list[dict[str, Any]], *, baseline: float | None = None,
         sigma: float | None = None, gap: dict[str, float] | None = None
         ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": status,
        "shot_time": stable_float(shot),
        "next_shot_time": stable_float(nxt) if nxt is not None else None,
        "window_start_index": None,
        "window_end_index": None,
        "window_start_time": None,
        "window_end_time": None,
        "endpoint_mode": None,
        "n_samples": 0,
        "heat_cal": None,
        "heat_uncertainty_cal": None,
        "heat_unit": "cal",
        "time_basis": "series_first_sample",
        "overlaps_next": False,
        "diagnostics": diags,
    }
    if baseline is not None:
        out["baseline_level"] = stable_float(baseline)
    if sigma is not None:
        out["baseline_sigma"] = stable_float(sigma)
    if gap is not None:
        out["gap"] = gap
    return out


def run_integration(power: dict[str, Any], schedule: dict[str, Any],
                    rules: RuleSet, *,
                    discarded: set[int] | None = None,
                    end_overrides: dict[int, float] | None = None,
                    run_label: str = "integration") -> dict[str, Any]:
    """Integrate every injection declared by the schedule against one power."""
    discarded = discarded or set()
    end_overrides = end_overrides or {}
    times = np.asarray(power["times"], dtype=float)
    powers = np.asarray(power["powers"], dtype=float)
    gaps = find_gaps(times, rules)

    shots: list[dict[str, Any]] = []
    for i, inj in enumerate(schedule["injections"]):
        tq = inj["time"].get("value")
        vq = inj["volume"].get("value")
        idx = inj.get("index") if inj.get("index") is not None else i + 1
        base_result: dict[str, Any] = {
            "injection": int(idx),
            "schedule_position": i + 1,
            "injected_volume_l": stable_float(vq) if vq is not None else None,
            "injected_volume_unit": "L",
            "shot_time_raw": inj["time"].get("raw"),
            "shot_time_s": stable_float(tq) if tq is not None else None,
            "discarded": int(idx) in discarded,
            "first_injection": i == 0,
        }
        if tq is None or vq is None:
            base_result.update(_bad(
                "incomplete_input", tq if tq is not None else 0.0, None,
                [diagnostic("incomplete_injection",
                            "schedule time/volume missing or unconvertible; "
                            "no default used", injection=int(idx),
                            time_diag=inj["time"].get("diagnostics"),
                            volume_diag=inj["volume"].get("diagnostics"))]))
            shots.append(base_result)
            continue
        next_t = None
        if i + 1 < len(schedule["injections"]):
            nt = schedule["injections"][i + 1]["time"].get("value")
            next_t = nt
        r = integrate_injection(
            times, powers, tq, next_t, rules, gaps=gaps,
            end_override=end_overrides.get(int(idx)))
        base_result.update(r)
        shots.append(base_result)

    n_ok = sum(1 for s in shots if s["status"] in ("ok", "clipped_overlap"))
    n_usable = sum(1 for s in shots
                   if s["status"] in ("ok", "clipped_overlap") and not s["discarded"])
    return {
        "kind": "integration_run",
        "label": run_label,
        "power_hash": power["content_hash"],
        "schedule_hash": schedule["content_hash"],
        "rule_version": rules.version,
        "gaps": [{"start": stable_float(b), "end": stable_float(a)}
                 for a, b in gaps],
        "discarded_first": sorted(int(s["injection"]) for s in shots
                                  if s["discarded"]),
        "end_overrides": {str(k): stable_float(v) for k, v in end_overrides.items()},
        "injections": shots,
        "n_ok": n_ok,
        "n_usable": n_usable,
        "basis": {
            "time_zero": "series_first_sample",
            "time_unit": "s",
            "power_unit": power.get("canonical_unit_power", "µcal/s"),
            "energy_unit": "cal",
        },
    }
