"""Injection integration: the hard rules live here, not in the browser.

- post-injection peaks are allowed to overlap: neighbouring peaks are split at
  the valley (midpoint fallback) and integrated separately;
- integration windows never span a data gap;
- a truncated recovery end is a semantic failure for that injection, not a
  silently extrapolated value;
- baseline subtraction uses the separately versioned baseline curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import Diagnostic, IntegrationPoint, IntegrationResult, stable_id
from .parser import baseline_view, gap_intervals, injections_view, power_view

RULES = {
    "name": "integration_rules",
    "version": 1,
    "baseline_band_w": 1.0e-7,        # |power| band considered "returned"
    "return_hold_samples": 4,         # samples inside band to call recovery
    "max_post_injection_s": 600.0,    # hard search horizon
    "method": "trapezoid",
}


@dataclass
class IntegrationParams:
    versioned: dict = field(default_factory=lambda: dict(RULES))


def _baseline_curve(power_material, baseline_material):
    if baseline_material is None:
        return None
    bt, bp = baseline_view(baseline_material)
    order = np.argsort(bt)
    bt, bp = bt[order], bp[order]
    return bt, bp


def baseline_power_at(t, baseline_curve):
    if baseline_curve is None:
        return 0.0
    bt, bp = baseline_curve
    return float(np.interp(t, bt, bp))


def _recovery_end(times, corrected, start_idx, end_bounds, gaps):
    band = RULES["baseline_band_w"]
    hold = RULES["return_hold_samples"]
    horizon = min(start_idx + 1, len(times) - 1)
    deadline_t = times[start_idx] + RULES["max_post_injection_s"]
    j = start_idx + 1
    held = 0
    while j < len(times):
        if times[j] > end_bounds:
            return None, "bounded"
        if times[j] > deadline_t:
            return None, "truncated"
        if not np.isnan(corrected[j]) and abs(corrected[j]) <= band:
            held += 1
            if held >= hold:
                return j, "recovered"
        else:
            held = 0
        j += 1
    return None, "truncated"


def _valley_split(times, corrected, a_idx, b_idx):
    """Index of the valley between two injection peaks (minimum abs excursion
    toward zero); midpoint fallback when the segment is unusable."""
    seg = corrected[a_idx:b_idx + 1]
    if len(seg) >= 3 and not np.any(np.isnan(seg)):
        local = int(np.argmin(np.abs(seg[1:-1]))) + 1
        return a_idx + local
    return a_idx + max(1, (b_idx - a_idx) // 2)


def integrate(power_material, injections_material, baseline_material,
              decision_ctx=None) -> IntegrationResult:
    decision_ctx = decision_ctx or {}
    times, powers, missing, pdiags = power_view(power_material)
    order = np.argsort(times)
    times, powers = times[order], powers[order]
    inv_order = order
    gaps = gap_intervals(times, [int(np.where(order == m)[0][0]) for m in missing])
    injections = injections_view(injections_material)
    bcurve = _baseline_curve(power_material, baseline_material)
    corrected = np.array([
        p if np.isnan(p) else p - baseline_power_at(t, bcurve)
        for t, p in zip(times, powers)
    ])

    n = len(injections)
    starts = [int(np.searchsorted(times, inj["time_s"], side="left")) for inj in injections]
    points: list[IntegrationPoint] = []
    diagnostics: list[dict] = []

    for k, inj in enumerate(injections):
        idx = starts[k]
        next_idx = starts[k + 1] if k + 1 < n else len(times)
        # Overlap split: if the previous peak has not recovered before this
        # injection, split at the valley; this injection's window starts there.
        split_s = None
        if k > 0:
            prev = points[k - 1]
            if prev.status in ("overlap_split",) or (
                    prev.window_end_s is not None and prev.window_end_s > inj["time_s"]):
                vi = _valley_split(times, corrected, starts[k - 1], idx)
                split_s = float(times[vi])
                window_start_idx = vi
            else:
                window_start_idx = idx
        else:
            window_start_idx = idx

        # end bound: valley with next injection if still non-recovered then,
        # otherwise full recovery.
        tentative_end, why = _recovery_end(times, corrected, window_start_idx,
                                           float(times[next_idx - 1]) if next_idx < len(times) else math.inf, gaps)
        if why == "bounded":
            vi = _valley_split(times, corrected, idx, next_idx - 1)
            end_idx = vi
            status = "overlap_split"
            split_s = float(times[vi])
        elif why == "truncated":
            end_idx = None
            status = "endpoint_truncated"
            diagnostics.append(Diagnostic(
                "endpoint_not_recovered",
                f"injection {inj['index']}: signal did not return to baseline "
                f"before samples ended; no heat is reported",
                {"injection_index": inj["index"]},
                "semantic_unresolvable").to_dict())
        else:
            end_idx = tentative_end
            status = "overlap_split" if k > 0 and points[k - 1].status == "overlap_split" else "ok"
            if k > 0 and points[k - 1].window_end_s is not None and points[k - 1].window_end_s > inj["time_s"]:
                status = "overlap_split"

        window_start_s = float(times[window_start_idx])
        window_end_s = None
        heat = None
        if end_idx is not None:
            window_end_s = float(times[end_idx])
            hit_gap = next((g for g in gaps
                            if g[0] >= window_start_s and g[1] <= window_end_s + 1e-12
                            and not math.isinf(g[1])), None)
            seg_t = times[window_start_idx:end_idx + 1]
            seg_p = corrected[window_start_idx:end_idx + 1]
            if hit_gap or np.any(np.isnan(seg_p)):
                status = "blocked_by_gap"
                loc = hit_gap if hit_gap else [window_start_s, window_end_s]
                diagnostics.append(Diagnostic(
                    "window_spans_gap",
                    f"injection {inj['index']}: integration window spans a data "
                    f"gap; heat is not reported",
                    {"injection_index": inj["index"], "gap_s": list(loc) if isinstance(loc, tuple) else loc},
                    "semantic_unresolvable").to_dict())
                window_end_s = float(times[end_idx])
            else:
                heat = float(np.trapezoid(seg_p, seg_t))

        points.append(IntegrationPoint(
            injection_index=inj["index"],
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            heat_j=heat,
            raw_heat_j=heat,
            status=status,
            diagnostics=[d for d in diagnostics
                         if d["location"].get("injection_index") == inj["index"]],
            split_boundary_s=split_s))

    drop_first = bool(decision_ctx.get("drop_first_injection"))
    if drop_first and points:
        points[0].status = "dropped"
        points[0].diagnostics = points[0].diagnostics + [Diagnostic(
            "first_injection_dropped",
            "first injection dropped by decision (pre-equilibration)",
            {"injection_index": points[0].injection_index}).to_dict()]

    input_summary = {
        "power_material": power_material.material_id,
        "injections_material": injections_material.material_id,
        "baseline_material": baseline_material.material_id if baseline_material else None,
        "n_samples": len(times),
        "n_injections": n,
        "missing_samples": missing,
        "drop_first_injection": drop_first,
        "time_reference": "seconds since acquisition start",
    }
    parents = [power_material.material_id, injections_material.material_id]
    if baseline_material:
        parents.append(baseline_material.material_id)
    result = IntegrationResult(
        result_id=stable_id("integ", [input_summary, RULES,
                                      sorted(parents),
                                      [(p.injection_index, p.status, round(p.heat_j or 0.0, 15))
                                       for p in points]]),
        power_version=power_material.version,
        injections_version=injections_material.version,
        baseline_version=baseline_material.version if baseline_material else 0,
        input_summary=input_summary,
        param_version=RULES["version"],
        parents=parents,
        points=[p.to_dict() for p in points],
        diagnostics=diagnostics + pdiags)
    return result


# ---------------------------------------------------------------------------
def apply_blank_correction(integration: IntegrationResult, blank_material,
                           rule: str, decisions: list[str]):
    """Dilution heat subtraction with an explicit correspondence rule.

    rule in {"by_index", "nearest_volume", "none"}. Missing correspondence is a
    positional semantic failure: raw heat stays, net heat is unresolved.
    """
    points = [dict(p) for p in integration.points]
    diagnostics = list(integration.diagnostics)
    net = {}
    if rule != "none" and blank_material is not None:
        blank = integrate(blank_material, _blank_injections_material(blank_material),
                          _blank_baseline_material(blank_material))
        bpoints = {(p["injection_index"]): p for p in blank.points}
        for p in points:
            idx = p["injection_index"]
            b = None
            if rule == "by_index":
                b = bpoints.get(idx)
            elif rule == "nearest_volume":
                cand = [q for q in blank.points if q["heat_j"] is not None]
                if cand:
                    b = min(cand, key=lambda q: abs(q["injection_index"] - idx))
            if b is None or b.get("heat_j") is None:
                p["blank_heat_j"] = None
                p["net_heat_j"] = None
                p["correspondence"] = "missing"
                diagnostics.append(Diagnostic(
                    "blank_correspondence_missing",
                    f"no {rule} blank injection corresponds to injection {idx}; "
                    f"dilution heat cannot be subtracted",
                    {"injection_index": idx, "rule": rule},
                    "semantic_unresolvable").to_dict())
            else:
                p["blank_heat_j"] = b["heat_j"]
                p["net_heat_j"] = (p["heat_j"] - b["heat_j"]) if p["heat_j"] is not None else None
                p["correspondence"] = f"{rule}:{b['injection_index']}"
                net[idx] = p["net_heat_j"]
    else:
        for p in points:
            p["blank_heat_j"] = None
            p["net_heat_j"] = p["heat_j"]
            p["correspondence"] = "none"
    out = IntegrationResult(**{**integration.to_dict(),
                               "points": points, "diagnostics": diagnostics})
    return out


def _blank_injections_material(blank_material):
    from .models import Material
    doc = blank_material.raw_representation
    return Material(material_id=blank_material.material_id + ":inj",
                    material_type="injections", version=blank_material.version,
                    raw_representation={"time_unit": doc.get("time_unit"),
                                        "volume_unit": doc.get("volume_unit"),
                                        "injections": doc.get("injections", [])})


def _blank_baseline_material(blank_material):
    doc = blank_material.raw_representation
    if doc.get("blank_baseline"):
        from .models import Material
        bd = doc["blank_baseline"]
        return Material(material_id=blank_material.material_id + ":bl",
                        material_type="baseline", version=blank_material.version,
                        raw_representation=bd)
    return None
