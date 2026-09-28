"""Injection integration.

Rules implemented here (machine-checkable, replayable):

* Post-injection peaks may overlap: integration windows are allowed to
  overlap in time, and each window is integrated independently.
* An integration window never crosses a data gap: the effective window is
  clipped to the surrounding valid sampling segment and the endpoint
  diagnostics record exactly what happened.
* Dilution-heat subtraction must state its correspondence rule
  (``blank_rule``) and only runs when the blank counterpart exists.
* First-injection discard is an explicit decision; discarded injections keep
  their raw integral but are excluded from fitting.
* Concentration revisions invalidate fits, never stored integrals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import SemanticUnresolved
from .ids import content_id

INTEGRATION_RULES_VERSION = "integration-rules/v1"


def _series(material: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pts = [p for p in material["payload"]["points"] if p.get("convertible")]
    t = np.array([p["time_s"] for p in pts], dtype=float)
    v = np.array([p["value"] for p in pts], dtype=float)
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _baseline_at(material: dict[str, Any], t: np.ndarray) -> np.ndarray:
    payload = material["payload"]
    if payload.get("mode") == "constant":
        const = payload.get("constant")
        if const is None:
            raise SemanticUnresolved(
                "baseline constant is unconvertible",
                diagnostics=[{"reason": "BASELINE_UNCONVERTIBLE"}],
            )
        return np.full_like(t, const["value"])
    bt, bv = _series(material)
    if bt.size == 0:
        raise SemanticUnresolved(
            "baseline curve has no convertible samples",
            diagnostics=[{"reason": "BASELINE_EMPTY"}],
        )
    return np.interp(t, bt, bv)


def _clip_to_segment(
    t: np.ndarray,
    window: tuple[float, float],
    gaps: list[dict[str, Any]],
) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    """Clip the window so it never crosses a data gap."""
    start, end = window
    diags: list[dict[str, Any]] = []
    for gap in gaps:
        g0, g1 = float(gap["start"]), float(gap["end"])
        if start < g0 and end > g1:
            # window spans the whole gap: keep the side containing the peak start
            end = g0
            diags.append(
                {
                    "reason": "WINDOW_CLIPPED_AT_GAP",
                    "gap": [g0, g1],
                    "kept_side": "before",
                }
            )
        elif g0 <= start < g1:
            start = g1
            diags.append({"reason": "WINDOW_START_MOVED_AFTER_GAP", "gap": [g0, g1]})
        elif g0 < end <= g1:
            end = g0
            diags.append({"reason": "WINDOW_END_MOVED_BEFORE_GAP", "gap": [g0, g1]})
    if end <= start:
        raise SemanticUnresolved(
            "integration window has no valid extent after gap clipping",
            diagnostics=diags + [{"reason": "EMPTY_WINDOW_AFTER_GAP_CLIP", "window": list(window)}],
        )
    return (start, end), diags


def integrate_injections(
    power: dict[str, Any],
    injections: dict[str, Any],
    baseline: dict[str, Any],
    *,
    window_rule: dict[str, Any],
    blank: dict[str, Any] | None,
    blank_power: dict[str, Any] | None,
    blank_injections: dict[str, Any] | None,
    blank_rule: dict[str, Any] | None,
    first_injection_discarded: bool,
    fit_budget: int | None = None,
) -> dict[str, Any]:
    """Integrate every injection and return a content-addressed result."""
    del fit_budget  # integration itself is O(n); budget governs model fitting
    t, p = _series(power)
    if t.size < 2:
        raise SemanticUnresolved(
            "power sampling has fewer than two convertible points",
            diagnostics=[{"reason": "INSUFFICIENT_POWER_POINTS", "material": power["id"]}],
        )
    gaps = list(power["payload"].get("gaps_s", []))
    base = _baseline_at(baseline, t)
    corrected = p - base

    entries = [e for e in injections["payload"]["entries"] if e.get("convertible")]
    entries.sort(key=lambda e: e["ordinal"])
    pre_s = float(window_rule.get("pre_s", 5.0))
    post_s = float(window_rule.get("post_s", 60.0))
    if pre_s < 0 or post_s <= 0:
        raise SemanticUnresolved(
            "window rule must have pre_s >= 0 and post_s > 0",
            diagnostics=[{"reason": "BAD_WINDOW_RULE", "rule": window_rule}],
        )

    blank_heats: dict[int, float] = {}
    blank_diags: list[dict[str, Any]] = []
    blank_used = False
    if blank is not None:
        if blank_rule is None or "correspondence" not in blank_rule:
            raise SemanticUnresolved(
                "dilution-heat subtraction requires an explicit correspondence rule",
                diagnostics=[{"reason": "BLANK_RULE_MISSING"}],
            )
        if blank_power is None or blank_injections is None:
            raise SemanticUnresolved(
                "blank material references missing power/schedule parts",
                diagnostics=[{"reason": "BLANK_PARTS_MISSING", "blank": blank["id"]}],
            )
        bt, bp = _series(blank_power)
        bbase = _baseline_at(baseline, bt)
        bcorr = bp - bbase
        blank_used = True
        for be in blank_injections["payload"]["entries"]:
            if not be.get("convertible"):
                continue
            win = (be["time_s"] - pre_s, be["time_s"] + post_s)
            win, _ = _clip_to_segment(bt, win, blank_power["payload"].get("gaps_s", []))
            mask = (bt >= win[0]) & (bt <= win[1])
            if mask.sum() >= 2:
                blank_heats[be["ordinal"]] = float(np.trapezoid(bcorr[mask], bt[mask]))

    per_injection: list[dict[str, Any]] = []
    for entry in entries:
        ordinal = entry["ordinal"]
        nominal = (entry["time_s"] - pre_s, entry["time_s"] + post_s)
        window, clip_diags = _clip_to_segment(t, nominal, gaps)
        mask = (t >= window[0]) & (t <= window[1])
        n_pts = int(mask.sum())
        endpoint_diags: list[dict[str, Any]] = list(clip_diags)
        if window[0] > nominal[0] + 1e-12:
            endpoint_diags.append({"reason": "START_CLIPPED", "nominal": nominal[0], "used": window[0]})
        if window[1] < nominal[1] - 1e-12:
            endpoint_diags.append({"reason": "END_CLIPPED", "nominal": nominal[1], "used": window[1]})
        if n_pts < 2:
            per_injection.append(
                {
                    "ordinal": ordinal,
                    "status": "SEMANTIC_UNRESOLVED",
                    "diagnostics": endpoint_diags
                    + [{"reason": "NO_SAMPLES_IN_WINDOW", "window": list(window)}],
                    "window_s": list(window),
                    "nominal_window_s": list(nominal),
                    "heat_J": None,
                    "net_heat_J": None,
                }
            )
            continue
        heat = float(np.trapezoid(corrected[mask], t[mask]))
        overlaps = [
            other["ordinal"]
            for other in entries
            if other["ordinal"] != ordinal
            and other["time_s"] - pre_s < nominal[1]
            and other["time_s"] + post_s > nominal[0]
        ]
        if overlaps:
            endpoint_diags.append({"reason": "OVERLAPPING_PEAK_WINDOWS", "with": overlaps})
        blank_heat = blank_heats.get(ordinal)
        blank_status = "not_requested"
        net = heat
        if blank_used:
            if blank_heat is None:
                blank_status = "missing_counterpart"
                endpoint_diags.append(
                    {"reason": "BLANK_COUNTERPART_MISSING", "ordinal": ordinal}
                )
            else:
                blank_status = "subtracted"
                net = heat - blank_heat
        per_injection.append(
            {
                "ordinal": ordinal,
                "status": "ok",
                "time_s": entry["time_s"],
                "volume_L": entry["volume_L"],
                "window_s": list(window),
                "nominal_window_s": list(nominal),
                "n_samples": n_pts,
                "heat_J": heat,
                "blank_heat_J": blank_heat,
                "blank_status": blank_status,
                "net_heat_J": net,
                "excluded_from_fit": bool(first_injection_discarded and ordinal == 1),
                "diagnostics": endpoint_diags,
            }
        )

    inputs_summary = {
        "power_material_id": power["id"],
        "injections_material_id": injections["id"],
        "baseline_material_id": baseline["id"],
        "blank_material_id": blank["id"] if blank else None,
        "blank_rule": blank_rule,
        "window_rule": {"pre_s": pre_s, "post_s": post_s},
        "first_injection_discarded": first_injection_discarded,
        "rules_version": INTEGRATION_RULES_VERSION,
    }
    result = {
        "kind": "integration",
        "inputs": inputs_summary,
        "per_injection": per_injection,
        "diagnostics": blank_diags,
        "unit": "J",
        "coordinate_basis": power["payload"].get("coordinate_basis", "elapsed seconds"),
    }
    result["id"] = content_id("int", result)
    return result
