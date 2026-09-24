"""Deterministic synthetic ITC traces used to author the offline case pack.

All traces are generated from a small number of physical parameters so the
case is genuinely a *titration thermogram* case (power sampling, injections,
baseline, dilution blank) rather than a generic table of numbers.

Power peaks are modelled as sharp injection spikes followed by exponential
relaxation back to baseline; overlapping peaks occur when the relaxation of
shot i has not decayed before shot i+1.
"""
from __future__ import annotations

import math
from typing import Sequence


def peak_shape(t: float, t0: float, area_cal: float, *,
               rise_s: float, decay_s: float) -> float:
    """Cal/s released at time t for one injection (exothermic -> positive)."""
    if t < t0:
        return 0.0
    dt = t - t0
    # normalised spike*exponential with analytic area 1:
    # amplitude chosen so integral == area_cal
    if rise_s <= 0:
        return 0.0
    amp = 1.0 / (rise_s * decay_s)  # integral of shape (1-e^-x/rise)*e^-x/decay
    shape = (math.exp(-dt / decay_s) - math.exp(-dt / rise_s - dt / decay_s))
    # simpler: rise as (1-exp(-dt/rise)) * exp(-dt/decay)
    shape = (1.0 - math.exp(-dt / rise_s)) * math.exp(-dt / decay_s)
    integral = decay_s - 1.0 / (1.0 / decay_s + 1.0 / rise_s)
    amp = 1.0 / integral
    return area_cal * amp * shape


def build_trace(injection_times: Sequence[float], areas_cal: Sequence[float],
                *, dt_s: float = 1.0, t_end_s: float | None = None,
                baseline_cal_s: float = 1.0e-6,
                rise_s: float = 2.0, decays_s: Sequence[float] | float = 12.0,
                gap: tuple[float, float] | None = None,
                seed_offset: float = 0.0) -> dict:
    """Return {times, powers, gap} sampled every ``dt_s`` seconds."""
    if isinstance(decays_s, (int, float)):
        decays_s = [float(decays_s)] * len(injection_times)
    if t_end_s is None:
        t_end_s = injection_times[-1] + 6.0 * float(decays_s[-1]) + 60.0
    n = int(t_end_s / dt_s) + 1
    times, powers = [], []
    for i in range(n):
        t = i * dt_s
        if gap is not None and gap[0] <= t < gap[1]:
            continue
        p = baseline_cal_s
        for t0, area, decay in zip(injection_times, areas_cal, decays_s):
            p += peak_shape(t, t0, area, rise_s=rise_s, decay_s=decay)
        times.append(t)
        # tiny deterministic ripple so "sampling" is visible but no RNG
        p += 2.0e-9 * math.sin((t + seed_offset) * 0.7)
        powers.append(p)
    return {"times": times, "powers": powers,
            "gap": list(gap) if gap else None,
            "baseline_cal_s": baseline_cal_s, "dt_s": dt_s}


def wiseman_heats(ratio_x: Sequence[float], n: float, k: float, c_m: float,
                  dh: float, v0_l: float) -> list[float]:
    """One-site expected net heats (cal) using the same quadratic as the fitter."""
    heats = []
    prev_qv = 0.0
    for i, x in enumerate(ratio_x):
        b = c_m * x + n * c_m + 1.0 / k
        disc = max(b * b - 4 * c_m * c_m * x * n, 0.0)
        theta = (b - math.sqrt(disc)) / (2 * n * c_m)
        theta = min(max(theta, 0.0), 1.0)
        qv = n * c_m * dh * theta
        heat = (qv - prev_qv) * v0_l
        heats.append(heat)
        prev_qv = qv
    return heats
