CANON_UNIT = {
    "time": "s",
    "power": "uW",
    "volume": "uL",
    "conc": "mM",
    "temp": "K",
    "cellvol": "mL",
    "energy": "uJ",
}

SCALE = {
    "time": {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0},
    "power": {"uW": 1.0, "uJ/s": 1.0, "mW": 1e3, "W": 1e6, "mcal/s": 4184.0},
    "volume": {"uL": 1.0, "nL": 1e-3, "mL": 1e3, "L": 1e6},
    "conc": {"nM": 1e-6, "uM": 1e-3, "mM": 1.0, "M": 1e3},
    "temp": {"K": 1.0},
    "cellvol": {"uL": 1e-3, "mL": 1.0, "L": 1e3},
    "energy": {"uJ": 1.0, "mJ": 1e3, "J": 1e6},
}


def convert_value(value, unit, kind, location):
    """Convert one raw value into the canonical unit of its kind.

    Returns (canonical_value_or_None, diagnostic_or_None). Unconvertible
    values are never dropped by callers; they stay in the raw layer and get
    a diagnostic attached at their original location.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, {
            "code": "unconvertible_value",
            "location": location,
            "raw": {"value": value, "unit": unit},
            "message": "value is not numeric; kept in raw layer",
        }
    if kind == "temp":
        if unit == "K":
            return float(value), None
        if unit == "degC":
            return float(value) + 273.15, None
        if unit == "degF":
            return (float(value) + 459.67) * 5.0 / 9.0, None
        return None, {
            "code": "unconvertible_unit",
            "location": location,
            "raw": {"value": value, "unit": unit},
            "message": "unknown temperature unit",
        }
    table = SCALE.get(kind, {})
    if unit in table:
        return float(value) * table[unit], None
    return None, {
        "code": "unconvertible_unit",
        "location": location,
        "raw": {"value": value, "unit": unit},
        "message": "unknown unit for kind %s" % kind,
    }
