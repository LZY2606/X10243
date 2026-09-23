"""Per-injection integration of ITC power traces.

Domain rules encoded here (each surfaced as a machine-readable diagnostic):
- post-injection peaks MAY overlap: consecutive integration windows are
  allowed to overlap; the fact is recorded as `overlaps_next`
- an integration window NEVER crosses a data gap: it is truncated at the
  gap edge (`window_truncated_at_gap`) or rejected (`window_starts_inside_gap`)
- the first injection may be discarded (status "discarded", never fitted)
- dilution-heat subtraction only happens under an explicit blank
  correspondence rule; a missing counterpart yields
  `blank_correspondence_missing` and the raw heat is kept (never faked)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import Ok, ResourceExhausted

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class InjectionInput:
    id: str
    index: int
    time_s: float
    volume_uL: float


def integrate_run(t, power, baseline_fn, injections, gaps, *,
                  pre_s, post_s, discard_first, snap_tol_s,
                  blank_heats=None, blank_rule_id=None,
                  max_samples=1_000_000):
    """Integrate every injection. Returns Ok(items) or ResourceExhausted."""
    t = np.asarray(t, dtype=float)
    power = np.asarray(power, dtype=float)
    used = 0
    items = []
    all_times = sorted(inj.time_s for inj in injections)
    for inj in injections:
        pos = f"injection[{inj.index}]:{inj.id}"
        if discard_first and inj.index == 0:
            items.append({
                "injection_id": inj.id, "index": inj.index,
                "window_s": [inj.time_s - pre_s, inj.time_s + post_s],
                "endpoints": None, "endpoint_times_s": None,
                "overlaps_next": False, "truncated_at_gap": False,
                "heat_uJ": None, "dilution_uJ": None, "net_heat_uJ": None,
                "status": "discarded",
                "diagnostics": [{
                    "code": "first_injection_discarded",
                    "message": "first injection discarded (priming/ diffusion artifact)",
                    "position": pos,
                }],
            })
            continue
        t0 = inj.time_s - pre_s
        t1 = inj.time_s + post_s
        overlaps_next = any(inj.time_s < ts < t1 for ts in all_times)
        diags = []
        truncated = False
        invalid = None
        for (g0, g1) in gaps:
            if t1 <= g0 or t0 >= g1:
                continue
            if t0 >= g0:
                invalid = "window_starts_inside_gap"
                diags.append({
                    "code": "window_starts_inside_gap",
                    "message": f"integration window starts inside data gap [{g0}, {g1}] s",
                    "position": pos,
                })
                break
            t1 = min(t1, g0)
            truncated = True
            diags.append({
                "code": "window_truncated_at_gap",
                "message": f"integration window truncated at data gap start {g0} s (window must not cross a gap)",
                "position": pos,
            })
        if invalid is None:
            mask = (t >= t0) & (t <= t1)
            used += int(mask.sum())
            if used > max_samples:
                return ResourceExhausted(
                    detail="integration sample budget exhausted",
                    budget=f"{max_samples} samples",
                )
            idx = np.flatnonzero(mask)
            if idx.size < 2:
                invalid = "window_empty"
                diags.append({
                    "code": "window_empty",
                    "message": "fewer than 2 samples inside integration window",
                    "position": pos,
                })
        if invalid is not None:
            items.append({
                "injection_id": inj.id, "index": inj.index,
                "window_s": [t0, t1], "endpoints": None, "endpoint_times_s": None,
                "overlaps_next": overlaps_next, "truncated_at_gap": truncated,
                "heat_uJ": None, "dilution_uJ": None, "net_heat_uJ": None,
                "status": "invalid", "diagnostics": diags,
            })
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        if abs(float(t[i0]) - t0) > snap_tol_s:
            diags.append({
                "code": "endpoint_snapped",
                "message": f"window start {t0}s snapped to sample {float(t[i0])}s",
                "position": f"{pos}.window.start",
            })
        if abs(float(t[i1]) - t1) > snap_tol_s:
            diags.append({
                "code": "endpoint_snapped",
                "message": f"window end {t1}s snapped to sample {float(t[i1])}s",
                "position": f"{pos}.window.end",
            })
        heat = float(_trapz(power[idx] - baseline_fn(t[idx]), t[idx]))
        dilution = None
        status = "ok"
        if blank_heats is not None:
            if inj.index in blank_heats:
                dilution = blank_heats[inj.index]
            else:
                status = "ok_no_blank"
                diags.append({
                    "code": "blank_correspondence_missing",
                    "message": f"blank rule '{blank_rule_id}' has no counterpart for injection index {inj.index}; dilution heat NOT subtracted",
                    "position": pos,
                })
        net = heat - (dilution if dilution is not None else 0.0)
        items.append({
            "injection_id": inj.id, "index": inj.index,
            "window_s": [t0, t1], "endpoints": [i0, i1],
            "endpoint_times_s": [float(t[i0]), float(t[i1])],
            "overlaps_next": overlaps_next, "truncated_at_gap": truncated,
            "heat_uJ": heat, "dilution_uJ": dilution, "net_heat_uJ": net,
            "status": status, "diagnostics": diags,
        })
    return Ok(items)
