"""Domain rules for ITC injection integration and binding-model candidates.

Three distinct failure outcomes are used everywhere, never conflated:
  * ``budget_exhausted`` - evaluation/resource budget ran out
  * ``corrupt_format``   - the input bytes/shape are malformed
  * ``unsolvable``       - well-formed input with no legal semantic answer

Key rules encoded here (and asserted in tests):
  * post-injection peaks may overlap; a window ends at the next injection
  * integration windows never cross a data gap (clipped or rejected)
  * dilution/blank subtraction requires an explicit correspondence rule
  * every fit records candidate parameters, SSE and residuals; all legal
    equally-good candidates are retained
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .units import convert

OK = "ok"
BUDGET_EXHAUSTED = "budget_exhausted"
CORRUPT_FORMAT = "corrupt_format"
UNSOLVABLE = "unsolvable"
STATUSES = (OK, BUDGET_EXHAUSTED, CORRUPT_FORMAT, UNSOLVABLE)


def diag(code, message, position=None, severity="error"):
    return {"code": code, "message": message, "position": position or {},
            "severity": severity}


@dataclass
class Outcome:
    status: str
    value: object = None
    diagnostics: list = field(default_factory=list)

    @property
    def ok(self):
        return self.status == OK


def baseline_at(knots, t):
    """Piecewise-linear baseline; constant extrapolation outside knots."""
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    for (t0, p0), (t1, p1) in zip(knots, knots[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return p0 + f * (p1 - p0)
    return knots[-1][1]


def integrate(trace, injections, knots, max_window_s, budget=200000):
    """Integrate baseline-corrected power per injection.

    ``trace``      : {"t": [s], "p": [uJ/s], "gaps": [(start_s, end_s), ...]}
    ``injections`` : [{"t": s, "v_L": L, "index": i}, ...]
    ``knots``      : sorted [(t_s, power_uJ/s), ...]
    """
    times = trace["t"]
    power = trace["p"]
    gaps = trace.get("gaps", [])
    results = []
    diagnostics = []
    used = 0
    for i, inj in enumerate(injections):
        t0 = inj["t"]
        t1 = t0 + max_window_s
        nxt = injections[i + 1]["t"] if i + 1 < len(injections) else None
        overlaps = False
        if nxt is not None and nxt < t1:
            t1 = nxt
            overlaps = True
        clipped = False
        inside_gap = None
        for gs, ge in gaps:
            if ge <= t0 or gs >= t1:
                continue
            if gs <= t0 < ge:
                inside_gap = (gs, ge)
                break
            t1 = min(t1, gs)
            clipped = True
        pos = {"injection_index": i, "window_start_s": t0}
        if inside_gap is not None:
            gs, ge = inside_gap
            diagnostics.append(diag(
                "injection_inside_gap",
                f"injection {i} at t={t0:g}s lies inside data gap "
                f"[{gs:g},{ge:g}]s; no integration window is legal", pos))
            results.append({"injection_index": i, "t0": t0, "t1": None,
                            "status": "invalid", "reason": "injection_inside_gap",
                            "flags": ["in_gap"]})
            continue
        idx = [k for k in range(len(times)) if t0 <= times[k] <= t1]
        used += len(idx)
        if used > budget:
            diagnostics.append(diag(
                "budget_exhausted",
                f"evaluation budget of {budget} sample evaluations exhausted "
                f"at injection {i}", pos))
            return Outcome(BUDGET_EXHAUSTED, None, diagnostics)
        if len(idx) < 2:
            diagnostics.append(diag(
                "insufficient_samples",
                f"injection {i} window [{t0:g},{t1:g}]s contains "
                f"{len(idx)} samples", pos))
            results.append({"injection_index": i, "t0": t0, "t1": t1,
                            "status": "invalid", "reason": "insufficient_samples",
                            "flags": ["window_clipped_by_gap"] if clipped else []})
            continue
        heat = 0.0
        for a, b in zip(idx, idx[1:]):
            dt = times[b] - times[a]
            pa = power[a] - baseline_at(knots, times[a])
            pb = power[b] - baseline_at(knots, times[b])
            heat += 0.5 * (pa + pb) * dt
        flags = []
        if overlaps:
            flags.append("overlaps_next")
        if clipped:
            flags.append("window_clipped_by_gap")
        results.append({"injection_index": i, "t0": t0, "t1": t1,
                        "status": "valid", "heat_uJ": heat,
                        "n_samples": len(idx), "flags": flags})
    return Outcome(OK, results, diagnostics)
