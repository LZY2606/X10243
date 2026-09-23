"""Dilution heat subtraction with an explicit correspondence rule."""
from __future__ import annotations

from typing import Any

from .integration import integrate_stream
from .models_common import make_value


def apply_blank_correction(sample_material, blank_material, *,
                           sample_integration, rule: dict[str, Any],
                           blank_integration=None) -> dict[str, Any]:
    """Match blank injections to sample injections and subtract areas.

    The correspondence rule is versioned and recorded in the output.  A sample
    injection without a legal counterpart is never silently zeroed: it is
    flagged ``blank_correspondence_missing`` and left uncorrected.
    """
    p = rule.get("parameters", rule)
    mode = p.get("correspondence", "injection_index")
    require_volume = bool(p.get("require_volume_match", True))
    on_missing = p.get("on_missing", "flag_uncorrected")
    time_window = 20.0
    if "time_within" in mode:
        digits = "".join(ch for ch in mode if ch.isdigit())
        time_window = float(digits) if digits else 20.0

    if blank_integration is None:
        blank_integration = integrate_stream(blank_material,
                                             _mirror_rule(sample_integration))

    blank_by_index = {
        w["injection_index"]: w for w in blank_integration["injections"]
    }

    corrected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for w in sample_integration["injections"]:
        idx = w["injection_index"]
        entry = {
            "injection_index": idx,
            "injection_time_s": w["injection_time_s"],
            "status": w["status"],
            "flags": list(w.get("flags", [])),
            "sample_area_uJ": w.get("area_uJ"),
            "blank_area_uJ": None,
            "corrected_area_uJ": None,
            "correspondence": None,
            "diagnostics": list(w.get("diagnostics", [])),
        }
        blank = blank_by_index.get(idx)
        correspondence = {"rule": mode, "matched_by": "injection_index",
                          "matched": False}
        if (blank is None and mode != "injection_index"
                and "time" in mode):
            blank = _match_by_time(blank_integration, w["injection_time_s"],
                                   time_window)
            if blank is not None:
                correspondence = {"rule": mode, "matched_by": "time_within",
                                   "time_window_s": time_window,
                                   "matched_blank_index":
                                       blank["injection_index"],
                                   "matched": True}
        elif blank is not None:
            correspondence["matched"] = True
            correspondence["matched_blank_index"] = blank["injection_index"]

        if w["status"] in ("discarded", "blocked"):
            entry["correspondence"] = correspondence
            corrected.append(entry)
            continue

        if blank is None or blank.get("status") in ("discarded", "blocked") \
                or blank.get("area_uJ") is None:
            entry["flags"].append("blank_correspondence_missing")
            entry["corrected_area_uJ"] = w.get("area_uJ")
            entry["uncorrected"] = True
            entry["diagnostics"].append({
                "code": "blank_correspondence_missing",
                "severity": "error" if on_missing == "reject" else "warning",
                "message": "No usable blank injection under the current "
                           "correspondence rule; heat left uncorrected.",
                "rule": mode,
            })
            if on_missing == "reject":
                entry["status"] = "uncorrected"
            diagnostics.append(entry["diagnostics"][-1])
            entry["correspondence"] = correspondence
            corrected.append(entry)
            continue

        if require_volume:
            sv = w["injection_volume"]["value"]
            bv = blank["injection_volume"]["value"]
            if abs(sv - bv) > 1e-9:
                entry["flags"].append("blank_volume_mismatch")
                entry["diagnostics"].append({
                    "code": "blank_volume_mismatch", "severity": "warning",
                    "sample_volume_uL": sv, "blank_volume_uL": bv,
                    "message": "Blank volume differs; scaling by volume ratio.",
                })
        sv = w["injection_volume"]["value"]
        bv = blank["injection_volume"]["value"]
        scale = sv / bv if require_volume and bv else 1.0
        blank_area = float(blank["area_uJ"]["value"]) * scale
        sample_area = float(w["area_uJ"]["value"])
        corrected_area = sample_area - blank_area
        entry["blank_area_uJ"] = make_value(blank_area, "uJ", "energy")
        entry["corrected_area_uJ"] = make_value(corrected_area, "uJ", "energy")
        entry["correspondence"] = correspondence
        corrected.append(entry)

    return {
        "kind": "blank_correction",
        "sample_integration_id": sample_integration.get("result_id"),
        "blank_material_id": blank_material["material_id"],
        "rule_id": rule.get("rule_id"),
        "rule_version": rule.get("rule_version"),
        "correspondence_rule": mode,
        "on_missing": on_missing,
        "injections": corrected,
        "diagnostics": diagnostics,
        "input_summary": {
            "n_sample": len(sample_integration["injections"]),
            "n_blank": len(blank_integration["injections"]),
            "n_missing": sum(1 for c in corrected
                             if "blank_correspondence_missing" in c["flags"]),
        },
    }


def _mirror_rule(integration_result: dict[str, Any]) -> dict[str, Any]:
    params = dict(integration_result.get("parameters", {}))
    return {
        "rule_set": "integration",
        "rule_version": integration_result.get("rule_version"),
        "parameters": params,
    }


def _match_by_time(blank_integration, t: float, window: float):
    best = None
    best_dt = None
    for w in blank_integration["injections"]:
        if w.get("status") in ("discarded", "blocked"):
            continue
        dt = abs(w["injection_time_s"] - t)
        if dt <= window and (best_dt is None or dt < best_dt):
            best = w
            best_dt = dt
    return best
