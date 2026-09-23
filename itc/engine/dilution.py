"""Dilution heat subtraction.

Blank correspondence is *always* explicit (rule ``blank_correspondence``).
Sample ordinal i maps to blank ordinal i+offset.  When no blank injection
exists at that ordinal, the sample injection keeps an unresolved dilution
component with diagnostics; it is never silently treated as zero dilution.
"""
from __future__ import annotations

import numpy as np

from ..errors import Diagnostic
from .hashing import sha12
from .integrate import integrate_all, _trace_arrays, _gaps, _first_gap_in


def blank_injection_areas(*, blank: dict[str, Any],
                          blank_injections: dict[str, Any],
                          baseline: dict[str, Any], rules: dict[str, Any]) -> dict:
    return integrate_all(power=blank, baseline=baseline,
                         injections_material=blank_injections, rules=rules)


def apply_dilution(*, sample: dict[str, Any], blank_run: dict | None,
                   rules: dict[str, Any]) -> dict[str, Any]:
    """sample: output of integrate_all() for sample run.
    blank_run: {areas: integrate_all output} or None when blank is absent."""
    corr_rules = rules["blank_correspondence"]
    offset = int(corr_rules.get("offset", 0))
    blank_by_ordinal: dict[int, dict] = {}
    blank_available: set[int] = set()
    if blank_run is not None:
        for b in blank_run["areas"]["integrals"]:
            blank_available.add(b["injection_ordinal"])
            if b.get("area_ucal") is not None:
                blank_by_ordinal[b["injection_ordinal"]] = b

    corrected: list[dict[str, Any]] = []
    for rec in sample["integrals"]:
        ordinal = rec["injection_ordinal"]
        target = ordinal + offset
        diags = list(rec.get("diagnostics", []))
        flags = list(rec.get("status_flags", []))
        dilution_area = None
        blank_ref = None
        status = "corrected"
        if rec.get("area_ucal") is None:
            status = "unresolved"
        elif target not in blank_available:
            diags.append(Diagnostic(
                "blank_correspondence_missing",
                f"no blank injection at ordinal {target} "
                f"(correspondence {corr_rules['mode']}, offset {offset}); "
                "dilution heat left unresolved, not assumed zero",
                target=f"injection:{ordinal}").to_dict())
            flags.append("blank_missing")
            status = "dilution_unresolved"
        elif target not in blank_by_ordinal:
            diags.append(Diagnostic(
                "blank_integral_unresolved",
                f"blank injection at ordinal {target} itself has no usable integral",
                target=f"injection:{ordinal}").to_dict())
            flags.append("blank_unresolved")
            status = "dilution_unresolved"
        else:
            br = blank_by_ordinal[target]
            dilution_area = float(br["area_ucal"])
            blank_ref = br["id"]

        net = None
        if rec.get("area_ucal") is not None and dilution_area is not None:
            net = float(rec["area_ucal"]) - dilution_area
        elif rec.get("area_ucal") is not None:
            status = "dilution_unresolved"

        cid = "dil_" + sha12({
            "sample_integral": rec["id"],
            "blank_integral": blank_ref,
            "offset": offset, "mode": corr_rules["mode"],
            "net_ucal": None if net is None else round(net, 6),
        })
        corrected.append({
            "id": cid,
            "injection_ordinal": ordinal,
            "sample_integral_id": rec["id"],
            "blank_integral_id": blank_ref,
            "blank_target_ordinal": target,
            "sample_area_ucal": rec.get("area_ucal"),
            "dilution_area_ucal": dilution_area,
            "net_heat_ucal": net,
            "discarded": rec.get("discarded", False),
            "status": status,
            "status_flags": flags,
            "diagnostics": diags,
            "correspondence": {"mode": corr_rules["mode"], "offset": offset},
            "rules_version": rules.get("version"),
        })
    return {"corrected": corrected,
            "correspondence": {"mode": corr_rules["mode"], "offset": offset},
            "blank_available_ordinals": sorted(blank_available)}
