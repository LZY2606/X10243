"""Blank correspondence and dilution-heat subtraction.

The correspondence *rule* is versioned and explicit. A missing or
volume-mismatched blank does not silently become zero: the sample injection is
reported with structured status ``blank_missing`` / ``blank_volume_mismatch``
and stays uncorrected (the reviewer chooses what to do).
"""
from __future__ import annotations

from .errors import Diagnostic
from .integration import integrate_injection
from .models import make_node

DEFAULT_CORRECTION = {
    "version": 1,
    "rule": "volume_group_ordinal",
    "volume_tolerance_l": 1e-9,   # 1 nL
}


def integrate_blank(blank_power, blank_schedule, baseline, integration_params):
    """Integrate every blank injection using the same window rules."""
    nodes = []
    for ordinal in range(len(blank_schedule["injections"])):
        nodes.append(integrate_injection(
            blank_power, blank_schedule, baseline, integration_params,
            ordinal))
    return nodes


def correspond(injection: dict, blanks: list[dict], rule: str,
               tol_l: float) -> dict:
    """Return {'blank': <injection dict>|None, 'status': ...}."""
    if rule == "ordinal":
        idx = injection["ordinal"]
        if idx >= len(blanks):
            return {"blank": None, "status": "blank_missing"}
        cand = blanks[idx]
        return _check_volume(injection, cand)
    if rule == "volume_group_ordinal":
        group = [b for b in blanks
                 if b["volume_l"] is not None and injection["volume_l"]
                 is not None and abs(b["volume_l"] - injection["volume_l"])
                 <= tol_l]
        if not group:
            return {"blank": None, "status": "blank_missing"}
        smaller = sorted(
            [b for b in blanks if b["volume_l"] is not None
             and injection["volume_l"] is not None
             and b["volume_l"] <= injection["volume_l"]],
            key=lambda b: (abs(b["volume_l"] - injection["volume_l"]),
                           b["ordinal"]))
        cand = smaller[0] if smaller else group[0]
        return _check_volume(injection, cand)
    return {"blank": None, "status": "correspondence_rule_unknown"}


def _check_volume(injection, cand) -> dict:
    tol = DEFAULT_CORRECTION["volume_tolerance_l"]
    if (injection["volume_l"] is None or cand["volume_l"] is None or
            abs(injection["volume_l"] - cand["volume_l"]) > tol):
        return {"blank": cand, "status": "blank_volume_mismatch"}
    return {"blank": cand, "status": "matched"}


def correct_run(sample_nodes, sample_schedule, blank_nodes, blank_schedule,
                blank_power, correction_params):
    """Produce one correction node per sample injection.

    Node parents are the sample integration and (when matched) the blank
    integration, so a blank revision changes exactly these derived layers.
    """
    rule = correction_params["rule"]
    tol = float(correction_params["volume_tolerance_l"])
    blank_injections = blank_schedule["injections"]
    blank_heat = {n.payload["ordinal"]: n.payload.get("heat_cal")
                  for n in blank_nodes}
    out = []
    for node in sample_nodes:
        ordinal = node.payload["ordinal"]
        inj = sample_schedule["injections"][ordinal]
        match = correspond({**inj, "ordinal": ordinal}, blank_injections,
                           rule, tol)
        parents = [node.id]
        diagnostics = []
        raw = node.payload.get("heat_cal")
        bheat = None
        corrected = None
        if node.payload["status"] not in (
                "integrated", "integrated_overlap_clip", "manual_endpoint"):
            status = f"sample_{node.payload['status']}"
        elif match["status"] != "matched":
            status = match["status"]
            diagnostics.append(Diagnostic(
                f"correction_{status}",
                f"injection #{ordinal}: {status.replace('_', ' ')} under "
                f"rule {rule!r}; no dilution subtraction applied", "warning",
                f"schedule:injection:{ordinal}",
                {"rule": rule, "injection_volume_raw": inj.get("volume_raw"),
                 "n_blank_injections": len(blank_injections)}))
        else:
            bord = match["blank"]["ordinal"]
            bheat = blank_heat.get(bord)
            if bheat is None or raw is None:
                status = "blank_integration_invalid"
            else:
                corrected = raw - bheat
                status = "corrected"
                parents.append(blank_nodes[bord].id)
        payload = {
            "ordinal": ordinal,
            "label": node.payload["label"],
            "status": status,
            "raw_heat_cal": raw,
            "blank_heat_cal": bheat,
            "corrected_heat_cal": corrected,
            "heat_unit": "cal",
            "rule": rule,
            "rule_version": correction_params["version"],
            "matched_blank_ordinal": (match["blank"]["ordinal"]
                                      if match["blank"] else None),
            "sample_status": node.payload["status"],
        }
        inputs = {
            "sample_ordinal": ordinal,
            "sample_heat_cal": raw,
            "rule": rule,
            "blank_schedule_n": len(blank_injections),
            "blank_checksum": _checksum(blank_power),
        }
        out.append(make_node(
            "correction", status, payload, inputs, parents,
            {"correction": int(correction_params["version"])}, diagnostics))
    return out


def _checksum(power) -> str:
    return f"{len(power['time_s'])}:{power['time_s'][0]}:{power['time_s'][-1]}"
