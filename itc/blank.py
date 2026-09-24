"""Blank control / dilution heat correspondence and subtraction.

The correspondence rule is explicit data (``by_index`` or ``by_volume``).
A sample injection without a matching blank is a named
``BLANK_CORRESPONDENCE_MISSING`` diagnostic: the sample heat is retained
un-subtracted rather than silently treated as zero.
"""

from __future__ import annotations

from .diagnostics import SEMANTIC_INVALID, diagnostic
from .hashing import digest
from .integrate import FIRST_INJECTION_DISCARDED
from .units import is_finite_q

BLANK_CORRESPONDENCE_MISSING = "BLANK_CORRESPONDENCE_MISSING"
BLANK_UNCONVERTIBLE = "BLANK_UNCONVERTIBLE"
BLANK_APPLIED = "BLANK_APPLIED"


def _usable_blank_rows(blank_material):
    out = []
    for b in blank_material.payload["injections"]:
        if (b["index"] is not None and is_finite_q(b["time"])
                and is_finite_q(b["volume"]) and is_finite_q(b["energy"])):
            out.append(b)
    return out


def correspond(records, blank_material, rules):
    """Pair each integrated injection with a blank injection per the rule."""
    diags = []
    rows = _usable_blank_rows(blank_material)
    if not rows:
        diags.append(diagnostic(SEMANTIC_INVALID,
                                "blank control has no usable injection rows"))
    correspondence = rules.get("correspondence", "by_index")
    required = rules.get("required", True)
    mapping = {}
    reasons = {}
    for rec in records:
        match, reason = _match_one(rec, rows, correspondence)
        reasons[rec["index"]] = reason
        mapping[rec["index"]] = match["index"] if match else None
    return mapping, reasons, diags, correspondence, required


def _match_one(rec, rows, correspondence):
    if correspondence == "by_index":
        for b in rows:
            if b["index"] == rec["index"]:
                return b, "same injection index"
        return None, f"no blank row with index {rec['index']}"
    if correspondence == "by_volume":
        best = None
        for b in rows:
            dv = abs(b["volume"]["value"] - rec["volume_l"])
            if best is None or dv < best[0]:
                best = (dv, b)
        if best is None:
            return None, "no blank rows available"
        return best[1], f"closest blank volume (|dv|={best[0] * 1e9:.3f} nL)"
    return None, f"unknown correspondence rule {correspondence!r}"


def apply_blank(integral_result, blank_material, rules):
    """Return corrected per-injection heats plus structured state."""
    records = integral_result["records"]
    mapping, reasons, diags, correspondence, required = correspond(
        records, blank_material, rules)
    by_index = {b["index"]: b for b in _usable_blank_rows(blank_material)}
    corrected = []
    for rec in records:
        flags = list(rec.get("flags", []))
        q_blank_j = None
        q_corr = rec["integral_j"]
        bidx = mapping.get(rec["index"])
        if FIRST_INJECTION_DISCARDED in flags or rec["integral_j"] is None:
            state = "SKIPPED"
        elif bidx is not None and rules.get("apply", True):
            q_blank_j = by_index[bidx]["energy"]["value"]
            q_corr = rec["integral_j"] - q_blank_j
            flags.append(BLANK_APPLIED)
            state = BLANK_APPLIED
        else:
            state = BLANK_CORRESPONDENCE_MISSING
            flags.append(BLANK_CORRESPONDENCE_MISSING)
            if required:
                msg = (f"injection {rec['index']}: {reasons.get(rec['index'])}; "
                       "heat retained without dilution correction")
                diags.append(diagnostic(SEMANTIC_INVALID, msg,
                                        injection=rec["index"]))
        corrected.append({
            "index": rec["index"],
            "integral_j": rec["integral_j"],
            "blank_index": bidx,
            "blank_heat_j": q_blank_j,
            "corrected_heat_j": q_corr,
            "correspondence_rule": correspondence,
            "correspondence_reason": reasons.get(rec["index"]),
            "state": state,
            "flags": flags,
            "usable_in_fit": rec.get("usable_in_fit", False) and state == BLANK_APPLIED,
        })
    result = {
        "corrected": corrected,
        "mapping": {str(k): v for k, v in mapping.items()},
        "reasons": {str(k): v for k, v in reasons.items()},
        "rule_id": digest("rule_blank_rules", rules),
        "parents": {"blank": blank_material.version_id,
                    "integral": integral_result["result_id"]},
        "diagnostics": diags,
    }
    result["result_id"] = digest("blank_correction", {
        "parents": {k: v for k, v in result["parents"].items() if v is not None},
        "rule_id": result["rule_id"],
        "corrected": [{"index": c["index"], "blank_index": c["blank_index"],
                       "integral_j": c["integral_j"], "blank_heat_j": c["blank_heat_j"],
                       "state": c["state"]} for c in corrected],
    })
    return result

