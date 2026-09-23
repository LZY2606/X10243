"""Unit / coordinate-basis handling.

Raw representations are preserved verbatim in material payloads.  The
derived layer normalizes into canonical units.  Values that cannot be
converted stay in place (as ``None`` in the derived array) and produce a
diagnostic carrying the position of the offending value.
"""
from __future__ import annotations

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "heat": "uJ",
}

_FACTORS = {
    "time": {"s": 1.0, "min": 60.0},
    "power": {"uJ/s": 1.0, "ucal/s": 4.184, "mJ/s": 1000.0},
    "volume": {"uL": 1.0, "mL": 1000.0, "nL": 0.001},
    "concentration": {"mM": 1.0, "uM": 0.001, "M": 1000.0},
    "heat": {"uJ": 1.0, "ucal": 4.184, "mJ": 1000.0},
}


def convert_scalar(quantity, value, unit):
    """Return canonical value or raise ValueError."""
    if quantity == "temperature":
        if unit == "K":
            return float(value)
        if unit in ("degC", "C"):
            return float(value) + 273.15
        raise ValueError(f"unknown temperature unit {unit!r}")
    table = _FACTORS.get(quantity)
    if table is None:
        raise ValueError(f"unknown quantity {quantity!r}")
    if unit not in table:
        raise ValueError(f"unknown unit {unit!r} for {quantity}")
    return float(value) * table[unit]


def normalize_series(quantity, values, unit, position_prefix, diagnostics):
    """Normalize a list of raw values.  Unconvertible entries stay as None
    in the derived array and produce a positioned diagnostic."""
    out = []
    for i, raw in enumerate(values):
        pos = f"{position_prefix}[{i}]"
        try:
            out.append(convert_scalar(quantity, raw, unit))
        except (TypeError, ValueError) as exc:
            out.append(None)
            diagnostics.append({
                "severity": "error",
                "position": pos,
                "code": "unit_unconvertible",
                "message": f"value {raw!r} ({unit}) not convertible to "
                           f"{CANONICAL[quantity]}: {exc}",
            })
    return out


def normalize_quantity(quantity, qobj, position, diagnostics):
    """Normalize a ``{"value":..,"unit":..,"basis":..}`` object."""
    if not isinstance(qobj, dict) or "value" not in qobj or "unit" not in qobj:
        diagnostics.append({
            "severity": "error", "position": position,
            "code": "quantity_malformed",
            "message": f"expected {{value,unit}} at {position}",
        })
        return None
    try:
        return convert_scalar(quantity, qobj["value"], qobj["unit"])
    except (TypeError, ValueError) as exc:
        diagnostics.append({
            "severity": "error", "position": position,
            "code": "unit_unconvertible",
            "message": f"value {qobj.get('value')!r} ({qobj.get('unit')}) "
                       f"not convertible: {exc}",
        })
        return None
