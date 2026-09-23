"""Blank / dilution-heat subtraction with an explicit correspondence rule.

The rule used is always recorded with the result. Injections whose blank
counterpart is missing keep their unsubtracted heat, are flagged, and
produce a located diagnostic -- subtraction is never faked.
"""

RULES = ("by_index", "by_volume", "none")


def apply_blank(integrals, blank_curve, rule):
    """integrals: result of integrate_injections. blank_curve:
    {"injections": [{"index","volume_uL","heat_uJ"}]}.
    rule: {"kind": "by_index"} | {"kind":"by_volume","tolerance_uL":x}
          | {"kind":"none"}.
    """
    kind = rule.get("kind")
    if kind not in RULES:
        raise ValueError(f"unknown blank rule {kind!r}")
    blanks = blank_curve.get("injections", [])
    diags = []
    out = []
    for rec in integrals["injections"]:
        r = dict(rec)
        r["blank_heat_uJ"] = None
        r["net_heat_uJ"] = None
        r["blank_applied"] = False
        r["blank_rule"] = kind
        if r["status"] != "ok" or kind == "none":
            out.append(r)
            continue
        match = None
        if kind == "by_index":
            match = next((b for b in blanks if b["index"] == r["index"]), None)
        elif kind == "by_volume":
            tol = float(rule.get("tolerance_uL", 0.5))
            cands = [b for b in blanks
                     if abs(b["volume_uL"] - r["volume_uL"]) <= tol]
            if cands:
                match = min(cands,
                            key=lambda b: (abs(b["volume_uL"] - r["volume_uL"]),
                                               b["index"]))
        if match is None:
            r["flags"] = list(r.get("flags", [])) + ["blank_missing"]
            diags.append({
                "code": "E_BLANK_MISSING",
                "message": f"no blank counterpart for injection {r['id']} "
                           f"under rule {kind}",
                "location": {"injection": r["id"], "rule": kind},
                "severity": "error"})
            out.append(r)
            continue
        r["blank_heat_uJ"] = match["heat_uJ"]
        r["blank_applied"] = True
        r["net_heat_uJ"] = r["heat_uJ"] - match["heat_uJ"]
        out.append(r)
    return {"injections": out,
            "diagnostics": integrals["diagnostics"] + diags,
            "params": integrals["params"], "blank_rule": rule,
            "gaps": integrals.get("gaps", [])}
