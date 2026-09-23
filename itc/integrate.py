"""Injection windowing and peak integration.

Business rules implemented here (kept server-side, testable without a browser):

R1  Post-injection peaks may overlap: an integration window is allowed to
    extend past the start of the next injection; the overlap is flagged, not
    forbidden.
R2  An integration window must never cross a data gap. If a gap intersects the
    nominal window, the window is truncated at the gap start and flagged
    ``window_truncated_by_gap``.
R3  Missing/invalid fields produce positioned diagnostics; nothing is filled
    with silent defaults.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .diagnostics import Diagnostic, Outcome, Status


def find_windows(
    injection_times: np.ndarray,
    trace_end: float,
    gaps: List[dict],
    width: float,
) -> List[dict]:
    """Compute one integration window per injection.

    `gaps` is a list of {"start": s, "end": e} intervals (canonical seconds).
    Returns a list of window dicts with flags.
    """
    windows = []
    n = len(injection_times)
    for i in range(n):
        t0 = float(injection_times[i])
        end = t0 + width
        flags: List[str] = []
        if i + 1 < n and end > float(injection_times[i + 1]):
            flags.append("overlaps_next")  # R1: allowed, recorded
        if end > trace_end:
            end = trace_end
            flags.append("clipped_at_trace_end")
        for gap in sorted(gaps, key=lambda g: g["start"]):
            gs, ge = float(gap["start"]), float(gap["end"])
            if t0 < gs < end:  # R2: never integrate across a gap
                end = gs
                flags.append("window_truncated_by_gap")
                break
            if gs <= t0 < ge:
                flags.append("starts_inside_gap")
        windows.append({
            "injection_index": i,
            "start": t0,
            "end": end,
            "flags": flags,
            "width": end - t0,
        })
    return windows


def integrate_trace(
    times: np.ndarray,
    power: np.ndarray,
    baseline_power: np.ndarray,
    windows: List[dict],
    excluded: Optional[set] = None,
) -> List[dict]:
    """Trapezoid integration of (power - baseline) over each window.

    Returns per-injection integral records. Excluded injections (e.g. the
    discarded first injection) keep their integral but are flagged so the
    fitting layer can skip them.
    """
    excluded = excluded or set()
    records = []
    for w in windows:
        mask = (times >= w["start"]) & (times <= w["end"])
        idx = np.nonzero(mask)[0]
        n_points = int(idx.size)
        if n_points >= 2:
            heat = float(np.trapezoid((power[idx] - baseline_power[idx]), times[idx]))
        else:
            heat = 0.0
        flags = list(w["flags"])
        if w["injection_index"] in excluded:
            flags.append("excluded_from_fit")
        if n_points < 2:
            flags.append("insufficient_samples")
        records.append({
            "injection_index": w["injection_index"],
            "window": {"start": w["start"], "end": w["end"]},
            "n_samples": n_points,
            "heat_uJ": heat,
            "flags": flags,
        })
    return records


def validate_injections(injections: List[dict], trace_start: float, trace_end: float) -> List[Diagnostic]:
    """Positioned validation of the injection schedule (R3)."""
    diags: List[Diagnostic] = []
    prev = None
    for i, inj in enumerate(injections):
        for field in ("time_s", "volume_uL"):
            if inj.get(field) is None:
                diags.append(Diagnostic(
                    code="field_missing",
                    message=f"injection {i} is missing required field {field!r}",
                    path=f"injections[{i}].{field}",
                ))
        t = inj.get("time_s")
        if t is not None:
            if not (trace_start <= t <= trace_end):
                diags.append(Diagnostic(
                    code="injection_out_of_range",
                    message=f"injection {i} at t={t}s lies outside the power trace [{trace_start}, {trace_end}]",
                    path=f"injections[{i}].time_s",
                ))
            if prev is not None and t <= prev:
                diags.append(Diagnostic(
                    code="injections_not_monotonic",
                    message=f"injection {i} at t={t}s does not follow the previous injection",
                    path=f"injections[{i}].time_s",
                    severity="warning",
                ))
            prev = t
        v = inj.get("volume_uL")
        if v is not None and v <= 0:
            diags.append(Diagnostic(
                code="volume_nonpositive",
                message=f"injection {i} has non-positive volume {v}",
                path=f"injections[{i}].volume_uL",
            ))
    return diags


def blank_correspondence(
    n_injections: int,
    blank_injections: List[dict],
    rule: str = "match_by_index",
) -> Outcome:
    """Map titration injections to blank (dilution) injections.

    The correspondence *rule* is part of the result so that dilution-heat
    subtraction can always state which rule was applied. Missing counterparts
    produce positioned diagnostics and those injections are left uncorrected
    (never silently zero-filled).
    """
    if rule != "match_by_index":
        return Outcome(Status.SEMANTIC_UNSOLVABLE, diagnostics=[Diagnostic(
            code="blank_rule_unknown",
            message=f"unsupported blank correspondence rule {rule!r}",
            path="blank.rule",
        )])
    mapping = {}
    diags: List[Diagnostic] = []
    for i in range(n_injections):
        if i < len(blank_injections):
            mapping[i] = i
        else:
            diags.append(Diagnostic(
                code="blank_correspondence_missing",
                message=f"no blank injection corresponds to titration injection {i} "
                        f"under rule 'match_by_index'; dilution heat not subtracted for it",
                path=f"injections[{i}]",
            ))
    return Outcome(
        Status.DIAGNOSTIC if diags else Status.OK,
        payload={"rule": rule, "mapping": mapping},
        diagnostics=diags,
    )
