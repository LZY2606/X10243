"""Dilution-heat (blank) subtraction.

The correspondence rule between sample injections and blank injections must
always be stated explicitly; the applied rule, the resulting mapping and any
missing correspondences are recorded as a machine-readable judgment event.
Injections without a blank counterpart are NOT silently zero-filled: they get
status blank_missing and no corrected heat.
"""
from __future__ import annotations

from .results import Result, OK, DIAGNOSTICS
from .units import Diagnostic

PARAM_VERSION = "blank-v1"
DEFAULT_RULE = {"match": "by_time", "time_tolerance_s": 5.0, "volume_tolerance_uL": 0.5}


def apply_blank(sample_integration, blank_integration, rule=None):
    """Returns (Result, events)."""
    rule = {**DEFAULT_RULE, **(rule or {})}
    diags = []
    rows = []
    mapping = {}
    missing = []
    blank_ok = [h for h in blank_integration["heats"] if h["status"] == "ok"]

    for h in sample_integration["heats"]:
        iid = h["injection_id"]
        if h["status"] != "ok":
            rows.append({
                "injection_id": iid, "status": "sample_invalid",
                "sample_heat_uJ": None, "blank_heat_uJ": None,
                "corrected_heat_uJ": None, "blank_injection_id": None,
            })
            continue
        match = None
        if rule["match"] == "by_time":
            cands = [b for b in blank_ok if abs(b["time"] - h["time"]) <= rule["time_tolerance_s"]]
            if cands:
                match = min(cands, key=lambda b: abs(b["time"] - h["time"]))
        elif rule["match"] == "by_volume":
            cands = [b for b in blank_ok if abs(b["volume_uL"] - h["volume_uL"]) <= rule["volume_tolerance_uL"]]
            if cands:
                match = min(cands, key=lambda b: abs(b["volume_uL"] - h["volume_uL"]))
        else:
            diags.append(Diagnostic("unknown_rule", f"unknown blank match rule {rule['match']!r}", "blank.rule", {"rule": rule}))
        if match is None:
            missing.append(iid)
            diags.append(Diagnostic(
                "blank_missing",
                f"no blank counterpart for {iid} under rule {rule['match']}",
                iid, {"rule": rule}))
            rows.append({
                "injection_id": iid, "status": "blank_missing",
                "sample_heat_uJ": h["heat_uJ"], "blank_heat_uJ": None,
                "corrected_heat_uJ": None, "blank_injection_id": None,
            })
        else:
            mapping[iid] = match["injection_id"]
            rows.append({
                "injection_id": iid, "status": "ok",
                "sample_heat_uJ": h["heat_uJ"], "blank_heat_uJ": match["heat_uJ"],
                "corrected_heat_uJ": h["heat_uJ"] - match["heat_uJ"],
                "blank_injection_id": match["injection_id"],
            })

    value = {
        "rows": rows,
        "rule": rule,
        "param_version": PARAM_VERSION,
        "units": {"heat": "uJ"},
    }
    events = [{"kind": "judgment.blank_rule", "payload": {
        "rule": rule, "mapping": mapping, "missing": missing,
    }}]
    return Result(DIAGNOSTICS if diags else OK, value, diags), events
