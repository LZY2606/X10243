"""Domain quantities and unit normalization.

Raw materials are stored verbatim (original string values and units). The
derived layer normalizes to SI-ish working units (s, L, M, uJ/s, K). Values
that cannot be converted are left in place in the raw material and reported
through diagnostics with a precise path; they are never silently dropped or
replaced with defaults.
"""

# scale units: unit -> (canonical_unit, multiply_by)
SCALE_UNITS = {
    "s": ("s", 1.0),
    "ms": ("s", 1e-3),
    "min": ("s", 60.0),
    "ucal/s": ("uJ/s", 4.184),
    "uJ/s": ("uJ/s", 1.0),
    "mcal/s": ("uJ/s", 4184.0),
    "uL": ("L", 1e-6),
    "mL": ("L", 1e-3),
    "L": ("L", 1.0),
    "nM": ("M", 1e-9),
    "uM": ("M", 1e-6),
    "mM": ("M", 1e-3),
    "M": ("M", 1.0),
    "kJ/mol": ("kJ/mol", 1.0),
    "kcal/mol": ("kJ/mol", 4.184),
}

# offset units: unit -> (canonical_unit, add_this)
OFFSET_UNITS = {
    "C": ("K", 273.15),
    "K": ("K", 0.0),
}


def normalize_quantity(raw, path):
    """Normalize {"value": ..., "unit": ...} to canonical units.

    Returns (value_or_None, canonical_unit_or_None, diagnostic_or_None).
    On failure the raw value stays where it was; the diagnostic carries the
    path so the caller can point at the exact position.
    """
    if not isinstance(raw, dict) or "value" not in raw or "unit" not in raw:
        return None, None, {
            "code": "quantity_malformed",
            "path": path,
            "message": "expected {'value': ..., 'unit': ...}",
            "severity": "error",
        }
    unit = raw["unit"]
    try:
        value = float(raw["value"])
    except (TypeError, ValueError):
        return None, None, {
            "code": "value_not_numeric",
            "path": path,
            "message": f"value {raw['value']!r} is not numeric; left in place",
            "severity": "error",
        }
    if unit in SCALE_UNITS:
        canon, factor = SCALE_UNITS[unit]
        return value * factor, canon, None
    if unit in OFFSET_UNITS:
        canon, offset = OFFSET_UNITS[unit]
        return value + offset, canon, None
    return None, None, {
        "code": "unit_unknown",
        "path": path,
        "message": f"unit {unit!r} has no conversion rule; value left in place",
        "severity": "error",
    }


def normalize_series(rows, value_key, unit, path):
    """Normalize a list of dict rows' numeric field; collect diagnostics."""
    out = []
    diags = []
    for i, row in enumerate(rows):
        try:
            out.append(float(row[value_key]))
        except (KeyError, TypeError, ValueError):
            out.append(None)
            diags.append({
                "code": "value_not_numeric",
                "path": f"{path}[{i}].{value_key}",
                "message": f"row {i} has non-numeric {value_key!r}; left in place",
                "severity": "error",
            })
    return out, diags
