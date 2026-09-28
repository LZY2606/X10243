"""Quantity handling. Source representations are retained at the material layer."""

from __future__ import annotations

from math import isfinite

UNITS = {
    "power": {"uW": 1.0, "µW": 1.0, "cal/s": 4.184, "J/s": 1_000_000.0},
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0, "minute": 60.0},
    "volume": {"uL": 1.0, "µL": 1.0, "mL": 1000.0, "L": 1_000_000.0},
    "temperature": {"K": "K", "C": "C", "degC": "C", "°C": "C"},
    "concentration": {"uM": 1.0, "µM": 1.0, "mM": 1000.0, "M": 1_000_000.0},
    "energy": {"uJ": 1.0, "µJ": 1.0, "mcal": 4.184, "cal": 4184.0, "J": 1_000_000.0},
}


def quantity(raw, kind, path):
    if isinstance(raw, dict):
        value = raw.get("value")
        unit = raw.get("unit")
        raw_repr = raw.get("raw", raw)
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        value, unit = raw
        raw_repr = list(raw)
    else:
        return None, [{
            "path": path,
            "code": "quantity_shape",
            "message": "quantity must be [value, unit] or {'value': ..., 'unit': ...}",
            "value": raw,
        }]
    diags = []
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        diags.append({"path": path + ".value", "code": "non_numeric", "message": "numeric finite value required", "value": value})
    allowed = UNITS[kind]
    if unit not in allowed:
        diags.append({"path": path + ".unit", "code": "unknown_unit", "message": f"unknown {kind} unit", "value": unit})
    if diags:
        return None, diags
    if kind == "temperature":
        canonical_unit = allowed[unit]
        canonical = float(value) if canonical_unit == "K" else float(value) + 273.15
    else:
        canonical = float(value) * allowed[unit]
    return {
        "value": float(value),
        "unit": unit,
        "raw": raw_repr,
        "canonical_value": canonical,
        "canonical_unit": "K" if kind == "temperature" else next(iter(allowed)),
    }, []


def canonical(raw, kind):
    parsed, diags = quantity(raw, kind, "$")
    if diags:
        raise ValueError(diags)
    return parsed["canonical_value"]
