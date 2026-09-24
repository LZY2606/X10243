"""Blank / dilution heat subtraction.

The correspondence rule is recorded explicitly with every corrected heat set.
When no matching blank injection exists, the corrected value is *not* silently
zeroed: the raw heat survives with ``blank_missing`` and the rule diagnostic.
"""
from __future__ import annotations

from typing import Any

CORRESPONDENCE_RULES = ("ordinal", "volume_matched", "none")


def subtract_blank(
    sample_run: dict[str, Any],
    blank_run: dict[str, Any] | None,
    concentrations: dict[str, Any],
    *,
    rule: str = "ordinal",
) -> dict[str, Any]:
    if rule not in CORRESPONDENCE_RULES:
        from .units import QuantityError

        raise QuantityError(
            "unknown-correspondence-rule",
            "blank_correction.rule",
            f"rule {rule!r} not in {sorted(CORRESPONDENCE_RULES)}",
            raw=rule,
        )
    blank_by_id = {}
    if blank_run is not None:
        for w in blank_run["windows"]:
            if rule == "volume_matched":
                key = round(float(w["volume_uL"]), 9)
            else:
                key = w["ordinal"]
            blank_by_id.setdefault(key, w)

    corrected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for w in sample_run["windows"]:
        entry: dict[str, Any] = {
            "window_id": w["window_id"],
            "injection_id": w["injection_id"],
            "index": w["index"],
            "ordinal": w["ordinal"],
            "status": w["status"],
            "raw_heat_uJ": w["heat_uJ"],
            "dilution_heat_uJ": None,
            "corrected_heat_uJ": None,
            "blank_window_id": None,
            "flags": list(w.get("flags", [])),
            "diagnostics": list(w.get("diagnostics", [])),
        }
        if w["status"] not in ("integrated", "integrated_open_endpoint"):
            corrected.append(entry)
            continue
        match_w = None
        if rule != "none" and blank_run is not None:
            key = round(float(w["volume_uL"]), 9) if rule == "volume_matched" else w["ordinal"]
            match_w = blank_by_id.get(key)
        if match_w is None or match_w["heat_uJ"] is None:
            entry["dilution_heat_uJ"] = None
            entry["corrected_heat_uJ"] = w["heat_uJ"]
            entry["flags"].append("blank_missing")
            diag = {
                "code": "blank-correspondence-missing",
                "where": f"injection[{w['index']}].blank",
                "message": (
                    "no blank injection corresponds under the declared rule; "
                    "raw heat retained without dilution subtraction"
                ),
                "rule": rule,
                "ordinal": w["ordinal"],
                "volume_uL": w["volume_uL"],
            }
            entry["diagnostics"].append(diag)
            diagnostics.append(diag)
        else:
            dilution = float(match_w["heat_uJ"])
            entry["dilution_heat_uJ"] = dilution
            entry["blank_window_id"] = match_w["window_id"]
            entry["corrected_heat_uJ"] = float(w["heat_uJ"]) - dilution
        corrected.append(entry)

    return {
        "kind": "corrected_heat_set",
        "correspondence_rule": rule,
        "sample_integration_id": sample_run.get("result_id"),
        "blank_integration_id": blank_run.get("result_id") if blank_run else None,
        "concentration_id": concentrations["material_id"],
        "concentration_revision": concentrations.get("revision", 1),
        "entries": corrected,
        "diagnostics": diagnostics,
    }
