"""Per-injection peak integration.

Hard rules exercised by the acceptance suite:

* Post-injection peaks MAY overlap: the integration window ends before the
  next injection (a guard), and the injection is marked ``PEAK_OVERLAP``.
* An integration window NEVER crosses a data gap; if a gap lies inside the
  window the result is ``WINDOW_CROSSES_GAP`` and the heat is unavailable.
* If the signal has not returned to baseline by the window end the endpoint is
  marked ``ENDPOINT_CLIPPED``.
* Baseline is subtracted from the separately versioned baseline curve (or a
  fitted local baseline) and the rule is recorded on every result.

First-injection discard is a *decision* handled upstream; integration is still
computed so the raw evidence remains inspectable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .parser import Injection, PowerSeries

# endpoint reason codes
ENDPOINT_RETURN = "RETURN_TO_BASELINE"
ENDPOINT_NEXT_INJECTION = "CLIPPED_BEFORE_NEXT_INJECTION"
ENDPOINT_SERIES_END = "SERIES_END"
ENDPOINT_GAP = "WINDOW_CROSSES_GAP"
ENDPOINT_TOO_FEW_POINTS = "TOO_FEW_POINTS"

MIN_POINTS_DEFAULT = 3
GUARD_BEFORE_NEXT_S_DEFAULT = 1.0


@dataclass
class IntegrationParams:
    version: str
    pre_injection_baseline_s: float = 10.0
    max_window_s: float = 180.0
    return_tolerance: float = 4.0        # multiples of baseline noise sigma
    min_points: int = MIN_POINTS_DEFAULT
    guard_before_next_s: float = GUARD_BEFORE_NEXT_S_DEFAULT
    baseline_rule: str = "REFERENCE_CURVE"  # or LOCAL_PRESHOT / ZERO


@dataclass
class IntegrationResult:
    injection_index: int
    start_time_s: float
    end_time_s: float
    n_points: int
    heat_ucal: float                      # NaN when not computable
    peak_power_ucal_s: float
    endpoint_reason: str
    overlap_next: bool
    crosses_gap: bool
    valid: bool
    diagnostics: list[dict] = field(default_factory=list)
    stable_id: str = ""

    def to_dict(self) -> dict:
        return {
            "stable_id": self.stable_id,
            "injection_index": self.injection_index,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "n_points": self.n_points,
            "heat_ucal": _json_num(self.heat_ucal),
            "peak_power_ucal_s": _json_num(self.peak_power_ucal_s),
            "endpoint_reason": self.endpoint_reason,
            "overlap_next": self.overlap_next,
            "crosses_gap": self.crosses_gap,
            "valid": self.valid,
            "diagnostics": self.diagnostics,
        }


def _json_num(x: float):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _arrays(series: PowerSeries) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([s.time_s for s in series.samples], dtype=float)
    p = np.array([s.power for s in series.samples], dtype=float)
    good = ~np.isnan(t)
    order = np.argsort(t[good], kind="stable")
    return t[good][order], p[good][order]


def _baseline_at(baseline: PowerSeries | None, t: np.ndarray,
                 shot: Injection, params: IntegrationParams
                 ) -> tuple[np.ndarray, float, str]:
    """Return baseline values aligned to *t* and the effective rule."""
    if baseline is not None and params.baseline_rule == "REFERENCE_CURVE":
        bt, bp = _arrays(baseline)
        finite = np.isfinite(bp)
        if finite.sum() >= 2:
            aligned = np.interp(t, bt[finite], bp[finite],
                                left=np.nan, right=np.nan)
            if np.isfinite(aligned).all():
                return aligned, float(np.nanstd(bp[:max(5, len(bp)//10)])), \
                    "REFERENCE_CURVE"
    # local pre-shot baseline: mean of power in [t0-pre, t0]
    return np.full_like(t, np.nan), float("nan"), "LOCAL_PRESHOT"


def integrate(power: PowerSeries, injections: list[Injection],
              params: IntegrationParams, baseline: PowerSeries | None = None,
              ) -> list[IntegrationResult]:
    t, p = _arrays(power)
    results: list[IntegrationResult] = []
    times = [inj.nominal_time_s for inj in injections]
    gaps = power.gaps

    for order_i, inj in enumerate(injections):
        t0 = inj.nominal_time_s
        if math.isnan(t0):
            results.append(_invalid(inj.index, math.nan, math.nan, 0,
                                    ENDPOINT_GAP,
                                    "injection nominal time is not convertible",
                                    "SCHEDULE_TIME_UNCONVERTIBLE"))
            continue
        t_start = t0
        next_t = (times[order_i + 1]
                  if order_i + 1 < len(injections)
                  and not math.isnan(times[order_i + 1]) else math.inf)

        # window end candidate limited by next injection guard
        hard_end = min(t[-1] if len(t) else t0,
                       next_t - params.guard_before_next_s,
                       t0 + params.max_window_s)
        overlap_next = (next_t - params.guard_before_next_s
                        < t0 + params.max_window_s) and math.isfinite(next_t)

        # gap check: any declared gap strictly inside (t0, hard_end]
        inside_gap = None
        for g in gaps:
            if t0 < g["end_time"] <= hard_end + 1e-9 and g["start_time"] >= t0 - 1e-9:
                inside_gap = g
                break

        seg_mask = (t >= t_start) & (t <= hard_end)
        ts = t[seg_mask]
        ps = p[seg_mask]

        if inside_gap is not None:
            results.append(_invalid(
                inj.index, t_start, hard_end, int(seg_mask.sum()),
                ENDPOINT_GAP,
                f"window [{t_start:.2f},{hard_end:.2f}] crosses data gap "
                f"({inside_gap['start_time']:.2f}->{inside_gap['end_time']:.2f})",
                "WINDOW_CROSSES_GAP",
                overlap_next=overlap_next))
            continue
        if len(ts) < params.min_points:
            results.append(_invalid(inj.index, t_start, hard_end, len(ts),
                                    ENDPOINT_TOO_FEW_POINTS,
                                    "fewer than minimum samples in window",
                                    "TOO_FEW_POINTS", overlap_next=overlap_next))
            continue

        base, sigma, rule = _baseline_at(baseline, ts, inj, params)
        if rule == "LOCAL_PRESHOT":
            pre = p[(t >= t0 - params.pre_injection_baseline_s) & (t < t0)]
            pre = pre[np.isfinite(pre)]
            if len(pre) >= 2:
                base = np.full_like(ts, float(np.mean(pre)))
                sigma = float(np.std(pre))
                rule = "LOCAL_PRESHOT"
            else:
                base = np.zeros_like(ts)
                sigma = 0.0
                rule = "ZERO"

        corrected = ps - base
        finite = np.isfinite(corrected)
        peak = float(np.nanmax(np.abs(corrected))) if finite.any() else math.nan

        # Detect return to baseline: after the peak the corrected signal must
        # stay within tolerance for a sustained interval (not merely touch it,
        # and not necessarily until the hard end - the next peak tail is
        # allowed to start later without poisoning this window's endpoint).
        tol = max(params.return_tolerance * (sigma if sigma > 0 else 1e-9),
                  1e-9)
        sustain_s = max(5.0, 4.0 * (power.median_dt_s or 1.0))
        end_idx = len(ts) - 1
        endpoint = ""
        peak_pos = int(np.argmax(np.abs(corrected))) if len(ts) else 0
        k = peak_pos
        while k < len(ts):
            k_hi = np.searchsorted(ts, ts[k] + sustain_s, side="right")
            if k_hi > k and np.all(np.abs(corrected[k:k_hi]) <= tol):
                end_idx = max(k_hi - 1, params.min_points - 1, peak_pos + 1)
                endpoint = ENDPOINT_RETURN
                overlap_next = False
                break
            k += 1
        if endpoint != ENDPOINT_RETURN:
            # window genuinely clipped: either by the next injection guard or
            # by the series end. overlap_next is then true only when the next
            # injection prevented this peak from completing.
            if math.isfinite(next_t) and hard_end <= next_t - \
                    params.guard_before_next_s + 1e-9 \
                    and next_t - params.guard_before_next_s <= t[-1]:
                endpoint = ENDPOINT_NEXT_INJECTION
                overlap_next = True
            else:
                endpoint = ENDPOINT_SERIES_END
                overlap_next = False

        tw = ts[: end_idx + 1]
        cw = corrected[: end_idx + 1]
        if len(tw) < params.min_points or not np.isfinite(cw).all():
            results.append(_invalid(inj.index, t_start, float(ts[end_idx]),
                                    len(tw), ENDPOINT_TOO_FEW_POINTS,
                                    "non-finite corrected samples in window",
                                    "NONFINITE_IN_WINDOW",
                                    overlap_next=overlap_next))
            continue
        heat = float(np.trapezoid(cw, tw)) if hasattr(np, "trapezoid") \
            else float(np.trapz(cw, tw))  # noqa: NPY201
        res = IntegrationResult(
            injection_index=inj.index,
            start_time_s=float(t_start),
            end_time_s=float(tw[-1]),
            n_points=len(tw),
            heat_ucal=heat,
            peak_power_ucal_s=peak,
            endpoint_reason=endpoint,
            overlap_next=bool(overlap_next),
            crosses_gap=False,
            valid=True,
            diagnostics=[{"baseline_rule": rule,
                          "noise_sigma": _json_num(sigma),
                          "window_clipped": endpoint != ENDPOINT_RETURN}],
        )
        if endpoint != ENDPOINT_RETURN:
            res.diagnostics.append({
                "code": "ENDPOINT_CLIPPED",
                "message": "signal not back to baseline at window end",
            })
        results.append(res)
    return results


def _invalid(index, start, end, n, reason, message, code,
             overlap_next=False) -> IntegrationResult:
    return IntegrationResult(
        injection_index=index, start_time_s=start, end_time_s=end,
        n_points=n, heat_ucal=math.nan, peak_power_ucal_s=math.nan,
        endpoint_reason=reason, overlap_next=overlap_next,
        crosses_gap=(reason == ENDPOINT_GAP), valid=False,
        diagnostics=[{"code": code, "message": message}])
