"""Shared numerical helpers and the stable value-object convention."""
from __future__ import annotations

from typing import Any
import math

import numpy as np


def make_value(value: float, display_unit: str, kind: str, *,
               coords: dict[str, Any] | None = None,
               basis: str | None = None, valid: bool = True,
               invalid_reason: str | None = None) -> dict[str, Any]:
    """A displayed value: numeric payload + unit/coordinate basis + validity.

    Every value rendered in the UI or returned by the API uses this shape so
    the front-end never has to guess units or a time origin.
    """
    v: dict[str, Any] = {
        "value": None if value is None or (isinstance(value, float)
                                           and (math.isnan(value)
                                                or math.isinf(value)))
        else float(value),
        "unit": display_unit,
        "kind": kind,
        "valid": bool(valid) and (value is not None),
    }
    if invalid_reason:
        v["invalid_reason"] = invalid_reason
    if coords:
        v["coords"] = coords
    if basis:
        v["basis"] = basis
    return v


def find_gaps(times: np.ndarray, *, gap_factor: float = 3.0
              ) -> list[dict[str, Any]]:
    """Detect structural gaps: an interval much longer than the median dt."""
    gaps: list[dict[str, Any]] = []
    if len(times) < 3:
        return gaps
    dts = np.diff(times)
    med = float(np.median(dts))
    if med <= 0:
        return gaps
    for i, d in enumerate(dts):
        if d > gap_factor * med:
            gaps.append({
                "end_prev_s": float(times[i]),
                "start_next_s": float(times[i + 1]),
                "missing_seconds": float(d - med),
            })
    return gaps


def estimate_tau(times: np.ndarray, signal: np.ndarray,
                 injections: list[dict[str, Any]], *, default: float = 9.0
                 ) -> float:
    """Estimate the common exponential decay time from early peak tails."""
    estimates = []
    for k, inj in enumerate(injections[:4]):
        t0 = float(inj["time_s"])
        next_t = (float(injections[k + 1]["time_s"])
                  if k + 1 < len(injections) else t0 + 60.0)
        m = (times >= t0) & (times < min(next_t, t0 + 40.0))
        t = times[m] - t0
        y = signal[m]
        m2 = (t >= 2.0) & (y > 0.05)
        if m2.sum() >= 5:
            try:
                slope, intercept = np.polyfit(t[m2], np.log(y[m2]), 1)
                if slope < 0:
                    estimates.append(-1.0 / slope)
            except (np.linalg.LinAlgError, ValueError):
                continue
    estimates = [e for e in estimates if 1.0 < e < 60.0]
    if not estimates:
        return default
    return float(np.median(estimates))


def syringe_concentration_profile(n_inj: int, volume_inj_uL: float,
                                  cell_volume_uL: float,
                                  ligand_uM_initial: float) -> np.ndarray:
    """Cumulative ligand concentration [uM] in the cell after each injection."""
    profile = np.zeros(n_inj)
    conc = 0.0
    r = volume_inj_uL / cell_volume_uL
    for i in range(n_inj):
        conc = conc + r * (ligand_uM_initial - conc)
        profile[i] = conc
    return profile
