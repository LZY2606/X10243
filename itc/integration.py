"""Per-injection integration with overlap and data-gap rules.

Hard domain rules encoded here:

* injection peaks are allowed to overlap: the *window* of injection *i* ends
  at ``min(t_start_i + width, t_start_(i+1))``;
* an integration window never crosses a data gap: it is clipped to the gap
  edge and the result is flagged ``window_clipped_by_gap``;
* the first injection is never silently deleted: discarding it for *fitting*
  is a separate, explicit decision (``discard_first``);
* blank/dilution subtraction must name its matching rule; unmatched shots
  stay uncorrected (``null`` q_corrected + diagnostic) instead of being
  faked with zero;
* baseline subtraction and blank subtraction are distinct stages with
  distinct evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from .identity import stable_id

INTEGRATION_RULE_VERSION = "integrate-1"
BASELINE_RULE_VERSION = "baseline-linear-1"
BLANK_RULE_V1 = "blank-match-ordinal-1"
BLANK_RULE_V2 = "blank-match-ordinal-volume-2"


@dataclass
class IntegrationParams:
    width_s: float = 180.0
    rule_version: str = INTEGRATION_RULE_VERSION
    baseline_rule_version: str = BASELINE_RULE_VERSION
    endpoint_overrides: dict[int, dict[str, float]] | None = None
    # endpoints in *canonical* seconds, validated against trace coverage


def _interp(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    return float(np.interp(x, xs, ys))


def _linear_baseline(points: list[dict[str, Any]]) -> Any:
    if not points:
        return None
    xs = np.array([p["time"]["canonical_value"] for p in points], dtype=float)
    ys = np.array([p["power"]["canonical_value"] for p in points], dtype=float)
    order = np.argsort(xs)
    return xs[order], ys[order]


def integrate_all(power_material: dict[str, Any],
                  params: IntegrationParams | None = None) -> dict[str, Any]:
    """Integrate every declared injection against one power material.

    Returns a deterministic dict containing one entry per injection plus
    run-level diagnostics.  No value is fabricated: clipped/uncovered
    windows are reported with statuses, not with extrapolated results.
    """
    params = params or IntegrationParams()
    injections = sorted(
        [i for i in power_material["items"]["injections"]],
        key=lambda it: it["index"])
    series = power_material["items"]["series"]
    gaps = power_material["items"].get("gaps", [])
    embedded_baseline = power_material["items"].get("baseline") or []

    xs = np.array([p["time"]["canonical_value"] for p in series], dtype=float)
    ps = np.array([p["power"]["canonical_value"] for p in series], dtype=float)
    base = _linear_baseline(embedded_baseline)

    t_min, t_max = (float(xs[0]), float(xs[-1])) if len(xs) else (0.0, 0.0)
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for pos, inj in enumerate(injections):
        idx = inj["index"]
        t0 = inj["time"]["canonical_value"]
        next_t0 = (injections[pos + 1]["time"]["canonical_value"]
                   if pos + 1 < len(injections) else None)

        override = (params.endpoint_overrides or {}).get(idx)
        start = t0
        if override is not None:
            if override["end"] <= start:
                diagnostics.append({
                    "code": "I050", "severity": "error",
                    "message": f"injection {idx}: endpoint override "
                               f"{override['end']:.6g}s must be after start "
                               f"{start:.6g}s",
                    "injection": idx, "line": inj["line"]})
                continue
            end = override["end"]
            override_used = True
        else:
            end = t0 + params.width_s
            override_used = False
        clipped_by_next = False
        if next_t0 is not None and end > next_t0:
            end = next_t0
            clipped_by_next = True

        gap_hits = []
        for gap in gaps:
            gs = gap["start"]["canonical_value"]
            ge = gap["end"]["canonical_value"]
            if start < ge and gs < end:  # interval overlap
                gap_hits.append(gap)
                if ge <= end:  # gap starts before window end -> clip end
                    end = min(end, gs)
        clipped_by_gap = bool(gap_hits)

        statuses: list[str] = []
        if start < t_min or end > t_max:
            statuses.append("window_extends_beyond_trace")
        if clipped_by_gap:
            statuses.append("window_clipped_by_gap")
        if clipped_by_next:
            statuses.append("window_clipped_by_next_injection")
        if override_used:
            statuses.append("endpoint_override")
        if end <= start:
            diagnostics.append({
                "code": "I051", "severity": "error",
                "message": f"injection {idx}: no integrable interval "
                           f"(start={start:.6g}, end={end:.6g})",
                "injection": idx, "line": inj["line"]})
            continue

        mask = (xs >= start) & (xs <= end)
        window_x = xs[mask]
        window_p = ps[mask]
        if len(window_x) < 2:
            statuses.append("insufficient_points")
            q_raw = q_baseline = 0.0
            residual_end = None
        else:
            p_corr = np.array(window_p, dtype=float)
            if base is not None:
                bvals = np.interp(window_x, base[0], base[1])
                p_corr = p_corr - bvals
            q_raw = float(_trapz(window_p, window_x))
            q_baseline = float(_trapz(p_corr, window_x))
            residual_end = float(p_corr[-1])

        # peak overlap evidence: a large residual at window end means the
        # current peak had not returned before the next shot
        overlap = bool(next_t0 is not None and residual_end is not None
                       and abs(residual_end) > 0.05 * max(1.0, abs(q_baseline) /
                                                          max(end - start, 1e-9)))
        if overlap:
            statuses.append("peak_overlaps_next_injection")

        entry = {
            "injection_index": idx,
            "ordinal": pos + 1,
            "source_line": inj["line"],
            "start_s": start,
            "end_s": end,
            "requested_width_s": None if override_used else params.width_s,
            "next_injection_start_s": next_t0,
            "volume_L": inj["volume"]["canonical_value"],
            "volume_raw": inj["volume"]["raw"],
            "time_raw": inj["time"]["raw"],
            "q_raw_J": q_raw,
            "q_baseline_J": q_baseline,
            "q_corrected_J": None,
            "blank_matched": False,
            "blank_match_rule": None,
            "blank_ref": None,
            "statuses": statuses,
            "gaps": [{"kind": g["kind"], "start_s": g["start"]["canonical_value"],
                      "end_s": g["end"]["canonical_value"], "line": g["line"],
                      "note": g.get("note")} for g in gap_hits],
            "rule_version": params.rule_version,
            "baseline_rule_version": params.baseline_rule_version,
            "n_samples": int(len(window_x)),
        }
        entry["integral_id"] = stable_id(
            "int", power_material.get("material_id"), idx, round(start, 9),
            round(end, 9), params.rule_version, params.baseline_rule_version)
        results.append(entry)

    diagnostics.sort(key=lambda d: (d.get("injection") or 0, d["code"]))
    return {
        "power_material_id": power_material.get("material_id"),
        "rule_version": params.rule_version,
        "baseline_rule_version": params.baseline_rule_version,
        "width_s": params.width_s,
        "integrals": results,
        "diagnostics": diagnostics,
    }


def match_blank(sample_integrals: list[dict[str, Any]],
                blank_integrals: list[dict[str, Any]],
                rule_version: str = BLANK_RULE_V1) -> list[dict[str, Any]]:
    """Apply a named blank/dilution matching rule.

    * ``blank-match-ordinal-1``: match by ordinal position only;
    * ``blank-match-ordinal-volume-2``: ordinal position then require |volume|
      within 5 % of the sample shot.

    Unmatched sample shots keep ``q_corrected_J = None`` and gain a
    positioned diagnostic - never zero subtraction.
    """
    out = [dict(e) for e in sample_integrals]
    blank_by_ordinal = {e["ordinal"]: e for e in blank_integrals}
    for entry in out:
        cand = blank_by_ordinal.get(entry["ordinal"])
        matched = False
        reason = None
        if cand is not None and rule_version == BLANK_RULE_V2:
            if entry["volume_L"] <= 0 or abs(cand["volume_L"] - entry["volume_L"]) \
                    > 0.05 * entry["volume_L"]:
                reason = ("blank shot volume does not match within 5% under "
                          f"{BLANK_RULE_V2}")
                cand = None
        if cand is not None:
            entry["q_corrected_J"] = (entry["q_baseline_J"]
                                      - cand["q_baseline_J"])
            entry["blank_matched"] = True
            entry["blank_match_rule"] = rule_version
            entry["blank_ref"] = cand["integral_id"]
            matched = True
        if not matched:
            entry["statuses"] = list(entry["statuses"]) + ["blank_unmatched"]
            entry["blank_match_rule"] = rule_version
            entry.setdefault("diagnostics", []).append({
                "code": "I060", "severity": "warning",
                "message": reason or f"no blank shot corresponds to ordinal "
                           f"{entry['ordinal']} under {rule_version}; "
                           "dilution heat left uncorrected",
                "injection": entry["injection_index"]})
    return out
