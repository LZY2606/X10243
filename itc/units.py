"""Unit / coordinate-basis handling.

Raw material keeps its original representation untouched. Normalization
happens only in the derived layer. Values that cannot be converted stay
in place and produce a located diagnostic; they are never silently
dropped or replaced with defaults.
"""

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "heat": "uJ",
    "cell_volume": "mL",
}

# Linear conversions value * factor -> canonical unit.
_LINEAR = {
    ("time", "s"): 1.0,
    ("time", "min"): 60.0,
    ("power", "uJ/s"): 1.0,
    ("power", "mJ/s"): 1.0e3,
    ("power", "ucal/s"): 4.184,
    ("power", "mcal/s"): 4.184e3,
    ("volume", "uL"): 1.0,
    ("volume", "nL"): 1.0e-3,
    ("concentration", "mM"): 1.0,
    ("concentration", "uM"): 1.0e-3,
    ("concentration", "nM"): 1.0e-6,
    ("concentration", "M"): 1.0e3,
    ("temperature", "K"): 1.0,
    ("heat", "uJ"): 1.0,
    ("heat", "mJ"): 1.0e3,
    ("heat", "ucal"): 4.184,
    ("cell_volume", "mL"): 1.0,
    ("cell_volume", "uL"): 1.0e-3,
}


def convert_scalar(value, from_unit, quantity):
    """Return canonical value. Raises KeyError for unknown units."""
    if quantity == "temperature" and from_unit == "C":
        return value + 273.15
    key = (quantity, from_unit)
    if key not in _LINEAR:
        raise KeyError(f"no conversion for {quantity}:{from_unit}")
    return value * _LINEAR[key]


def normalize_series(values, from_unit, quantity, location):
    """Normalize a list of scalars.

    Returns (canonical_values, diagnostics). Unconvertible entries keep
    their raw value in place (None slot is NOT invented) and each gets a
    located diagnostic.
    """
    out, diags = [], []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            diags.append({
                "code": "E_VALUE_MISSING",
                "message": f"missing value at {location}[{i}]",
                "location": {"series": location, "index": i},
                "severity": "warning",
            })
            continue
        try:
            out.append(convert_scalar(float(v), from_unit, quantity))
        except (KeyError, ValueError, TypeError) as exc:
            out.append(v)  # raw value stays in place
            diags.append({
                "code": "E_UNIT_UNKNOWN",
                "message": f"cannot convert {v!r} ({quantity}/{from_unit}) at {location}[{i}]: {exc}",
                "location": {"series": location, "index": i},
                "severity": "error",
            })
    return out, diags
