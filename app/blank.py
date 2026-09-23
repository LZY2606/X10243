"""Blank (dilution heat) correspondence and per-injection normalisation."""

from __future__ import annotations

AVOGADRO_UNUSED = None
CAL4 = 4.184


def match_blanks(sample_integrals, blank_integrals, rule):
    """Pair sample injections to blank injections.

    rule: {"mode": "ordinal"} or {"mode": "volume", "tolerance_fraction": x}
    Returns mapping ordinal -> blank integral plus per-injection diagnostics.
    A missing counterpart is reported; it never becomes zero heat.
    """
    mode = rule.get("mode", "ordinal")
    by_ordinal = {b["ordinal"]: b for b in blank_integrals}
    matched, diagnostics = {}, []
    for r in sample_integrals:
        ordn = r["ordinal"]
        if mode == "ordinal":
            b = by_ordinal.get(ordn)
            if b is None:
                diagnostics.append(_missing(ordn, rule, f"no blank injection with ordinal {ordn}"))
                continue
        elif mode == "volume":
            tol = float(rule.get("tolerance_fraction", 0.02))
            b = _volume_match(r, blank_integrals, tol)
            if b is None:
                diagnostics.append(_missing(
                    ordn, rule, f"no blank injection volume within {tol:.0%}"))
                continue
        else:
            diagnostics.append({"code": "UNKNOWN_BLANK_RULE", "ordinal": ordn,
                                "rule": rule, "message": f"unknown correspondence mode {mode}"})
            continue
        if b["energy_j"] is None:
            diagnostics.append(_missing(ordn, rule,
                                         f"blank ordinal {b['ordinal']} has no usable integral"))
            continue
        matched[ordn] = b
    return matched, diagnostics


def _volume_match(r, blanks, tol):
    if r.get("volume_l") is None:
        return None
    best, best_diff = None, None
    for b in blanks:
        if b.get("volume_l") in (None, 0) or b["energy_j"] is None:
            continue
        diff = abs(b["volume_l"] - r["volume_l"]) / max(r["volume_l"], 1e-30)
        if diff <= tol and (best_diff is None or diff < best_diff):
            best, best_diff = b, diff
    return best


def _missing(ordn, rule, message):
    return {"code": "BLANK_CORRESPONDENCE_MISSING", "ordinal": ordn,
            "rule": rule, "message": message}


def normalize(sample_integrals, matched_blanks, concentrations, cell_volume_l,
              *, blank_series_id=None):
    """Build per-injection normalized heats (J per mol injectant).

    Concentration revisions only change this layer (and downstream models);
    the original power integrals are never altered.
    """
    ligand_c = (concentrations or {}).get("ligand", {}).get("value_si")
    macro_c = (concentrations or {}).get("macromolecule", {}).get("value_si")
    rows, diagnostics = [], []
    if ligand_c in (None, 0):
        diagnostics.append({"code": "CONCENTRATION_UNAVAILABLE", "role": "ligand",
                            "message": "ligand concentration missing; normalization withheld"})
    if cell_volume_l in (None, 0):
        diagnostics.append({"code": "CELL_VOLUME_UNAVAILABLE",
                            "message": "cell volume missing; normalization withheld"})
    for r in sample_integrals:
        ordn = r["ordinal"]
        raw_j = r["energy_j"]
        blank_j = None
        if ordn in matched_blanks:
            blank_j = matched_blanks[ordn]["energy_j"]
        elif raw_j is not None:
            diagnostics.append({**next(
                (d for d in r.get("diagnostics", []) if d.get("code") == "BLANK_MISSING"),
                {"code": "BLANK_CORRESPONDENCE_MISSING", "ordinal": ordn,
                 "message": "blank counterpart missing; corrected heat withheld"})})
        corrected_j = None if raw_j is None or blank_j is None else raw_j - blank_j
        q_ucal = None if corrected_j is None else corrected_j / CAL4 * 1e6
        moles = None
        q_per_mol_j = None
        q_per_mol_kcal = None
        if ligand_c and r.get("volume_l") and corrected_j is not None:
            moles = ligand_c * r["volume_l"]
            if moles > 0:
                q_per_mol_j = corrected_j / moles
                q_per_mol_kcal = q_per_mol_j / CAL4 / 1000.0
        rows.append({
            "ordinal": ordn,
            "volume_l": r.get("volume_l"),
            "excluded": bool(r.get("excluded", False)),
            "raw_energy_j": raw_j,
            "blank_energy_j": blank_j,
            "blank_ordinal": matched_blanks[ordn]["ordinal"] if ordn in matched_blanks else None,
            "blank_series_id": blank_series_id,
            "corrected_energy_j": corrected_j,
            "corrected_energy_ucal": q_ucal,
            "injected_mol": moles,
            "q_per_mol_j": q_per_mol_j,
            "q_per_mol_kcal": q_per_mol_kcal,
            "usable": (not r.get("excluded", False)) and corrected_j is not None
                      and q_per_mol_j is not None,
        })
    return {
        "rows": rows,
        "ligand_concentration_m": ligand_c,
        "macromolecule_concentration_m": macro_c,
        "cell_volume_l": cell_volume_l,
        "diagnostics": diagnostics,
    }
