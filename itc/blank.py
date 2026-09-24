"""Blank / dilution-heat handling.

Rule encoded here: dilution-heat subtraction MUST name its correspondence
rule (e.g. by_index against a versioned blank run). Where the blank run has
no counterpart for a sample injection, no value is invented: the injection
keeps every legitimate dilution candidate and a positioned diagnostic.
"""
from __future__ import annotations

RULE_BY_INDEX = "by_index"


def blank_correspondence(sample_integrals, blank_integrals, rule=None):
    """Map blank heats onto sample injections under an explicit rule."""
    rule = rule or {"kind": RULE_BY_INDEX}
    if rule.get("kind") != RULE_BY_INDEX:
        raise ValueError(f"unknown blank correspondence rule {rule}")
    by_index = {b["index"]: b for b in blank_integrals}
    out, diags = [], []
    for s in sample_integrals:
        b = by_index.get(s["index"])
        if b is None or b["heat_uJ"] is None:
            diags.append({
                "severity": "warning", "code": "BLANK_MISSING",
                "message": f"no blank counterpart for injection {s['index']} "
                           f"under rule 'by_index'; dilution candidates kept",
                "location": {"injection_id": s["injection_id"]},
                "entity_ids": [s["injection_id"]]})
            out.append({"injection_id": s["injection_id"], "index": s["index"],
                        "blank_heat_uJ": None})
        else:
            out.append({"injection_id": s["injection_id"], "index": s["index"],
                        "blank_heat_uJ": b["heat_uJ"]})
    return out, diags


def dilution_candidates(sample_heats, k=4):
    """Legitimate dilution estimates when no blank correspondence exists.

    Returns all candidates (none / tail-mean / tail-linear); the caller keeps
    every one of them because the choice is not uniquely decidable.
    """
    valid = [(i, h) for i, h in enumerate(sample_heats) if h is not None]
    cands = [{"kind": "none", "value_uJ": 0.0}]
    if len(valid) >= 2:
        tail = valid[-k:] if len(valid) >= k else valid
        mean = sum(h for _, h in tail) / len(tail)
        cands.append({"kind": "tail_mean", "value_uJ": mean,
                      "from_indices": [i for i, _ in tail]})
        if len(tail) >= 2:
            n = len(tail)
            sx = sum(i for i, _ in tail)
            sy = sum(h for _, h in tail)
            sxx = sum(i * i for i, _ in tail)
            sxy = sum(i * h for i, h in tail)
            denom = n * sxx - sx * sx
            if denom != 0:
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
                cands.append({"kind": "tail_linear",
                              "slope_uJ_per_inj": slope,
                              "intercept_uJ": intercept,
                              "from_indices": [i for i, _ in tail]})
    return cands
