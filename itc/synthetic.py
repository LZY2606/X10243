"""Deterministic titration signal generator for the bundled offline case.

Not part of the production pipeline; used by scripts/build_case.py to create
real domain material files (power sampling, injections, baseline, blank).
"""

from __future__ import annotations

import math

import numpy as np


def _peak(t, center, amplitude_uw, tau_s, rise_s=2.0):
    """Rise then exponential-decay ITC peak in micro-watts."""
    x = t - center
    out = np.zeros_like(x, dtype=float)
    rising = (x >= 0) & (x <= rise_s)
    out[rising] = amplitude_uw * (x[rising] / rise_s)
    falling = x > rise_s
    out[falling] = amplitude_uw * np.exp(-(x[falling] - rise_s) / tau_s)
    return out


def one_site_heats(volume_inj_l, v0_l, c0_lab_m, x0_lab_m, ka, dh, n=1.0):
    """Wiseman one-site heat per injection (J), exact cell recurrence.

    Cell concentration ``c_i`` (macromolecule) and ligand ``x_i`` after i
    injections follow the displaced-volume recurrence.
    """
    theta = (1.0 + x0_lab_m / c0_lab_m + 1.0 / (ka * c0_lab_m)) / 2.0 - np.sqrt(
        ((1.0 + x0_lab_m / c0_lab_m + 1.0 / (ka * c0_lab_m)) / 2.0) ** 2
        - x0_lab_m / c0_lab_m
    )
    q_i = n * c0_lab_m * dh * v0_l * theta / (1.0 + 0.5 * volume_inj_l / v0_l)
    return q_i
