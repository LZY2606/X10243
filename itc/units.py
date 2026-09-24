"""Unit handling.

Raw values are preserved exactly as imported. Normalization to canonical
units happens only in the derived layer. Values that cannot be converted
stay in their original position and carry a diagnostic; they are never
silently dropped or replaced with defaults.
"""

# Linear conversions: value_to = value_from * factor
LINEAR = {
    ("uJ", "J"): 1e-6, ("mJ", "J"): 1e-3, ("J", "J"): 1.0,
    ("uJ/s", "J/s"): 1e-6, ("uW", "J/s"): 1e-6, ("mW", "J/s"): 1e-3,
    ("J/s", "J/s"): 1.0, ("ucal/s", "J/s"): 4.184e-6,
    ("uL", "L"): 1e-6, ("mL", "L"): 1e-3, ("L", "L"): 1.0,
    ("s", "s"): 1.0, ("min", "s"): 60.0, ("h", "s"): 3600.0,
    ("uM", "M"): 1e-6, ("mM", "M"): 1e-3, ("M", "M"): 1.0,
    ("K", "K"): 1.0,
    ("kcal/mol", "J/mol"): 4184.0, ("J/mol", "J/mol"): 1.0,
    ("kJ/mol", "J/mol"): 1000.0,
}

# Affine conversions: value_to = value_from * scale + offset
AFFINE = {
    ("degC", "K"): (1.0, 273.15),
}

CANONICAL = {
    "time": "s",
    "power": "J/s",
    "heat": "J",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
    "molar_enthalpy": "J/mol",
}


def normalize(value, from_unit, to_unit):
    """Return (converted_value, diagnostic_or_None). Never raises."""
    if from_unit == to_unit:
        return value, None
    if (from_unit, to_unit) in LINEAR:
        return value * LINEAR[(from_unit, to_unit)], None
    if (from_unit, to_unit) in AFFINE:
        scale, offset = AFFINE[(from_unit, to_unit)]
        return value * scale + offset, None
    return value, {
        "code": "unit_unconvertible",
        "from": from_unit,
        "to": to_unit,
        "value": value,
    }


def factor(from_unit, to_unit):
    """Linear scale factor between units, or (None, diagnostic)."""
    if from_unit == to_unit:
        return 1.0, None
    if (from_unit, to_unit) in LINEAR:
        return LINEAR[(from_unit, to_unit)], None
    return None, {
        "code": "unit_unconvertible",
        "from": from_unit,
        "to": to_unit,
    }
