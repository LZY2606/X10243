"""Unit layer.

Raw quantities are stored exactly as imported (raw value + raw unit + basis).
Normalization happens only here, in the derived layer. Values that cannot be
converted stay in place (raw payload untouched) and produce a positioned
diagnostic; nothing is silently dropped for the sake of plotting.
"""
from __future__ import annotations

CANONICAL_UNIT = {
    "time": "s",
    "volume": "uL",
    "concentration": "mM",
    "power": "uJ/s",
    "heat": "uJ",
    "temperature": "K",
}

_LINEAR = {
    "time": {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0},
    "volume": {"ul": 1.0, "ml": 1.0e3, "nl": 1e-3},
    "concentration": {"mm": 1.0, "um": 1e-3, "nm": 1e-6, "m": 1e3},
    "power": {"uj/s": 1.0, "ucal/s": 4.184, "mj/s": 1e3, "mcal/s": 4.184e3},
    "heat": {"uj": 1.0, "ucal": 4.184, "mj": 1e3, "j": 1e6, "kj": 1e9, "kcal": 4.184e6},
}


def _canon_unit(raw: str) -> str:
    return (raw or "").strip().replace("µ", "u").replace("μ", "u").replace("°", "deg")


def normalize_quantity(raw_value, raw_unit, kind: str, location=None):
    """Return (value_in_canonical_unit, canonical_unit, diagnostics).

    On failure value is None and a diagnostic explains why; callers keep the
    raw representation in place.
    """
    loc = location or {}
    diags = []
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        diags.append({
            "severity": "error", "code": "VALUE_NOT_NUMERIC",
            "message": f"value {raw_value!r} is not numeric",
            "location": loc,
        })
        return None, CANONICAL_UNIT.get(kind), diags

    unit = _canon_unit(raw_unit)
    if kind == "temperature":
        if unit.lower() == "k":
            return value, "K", diags
        if unit.lower() in ("c", "degc"):
            return value + 273.15, "K", diags
        diags.append({
            "severity": "error", "code": "UNIT_UNCONVERTIBLE",
            "message": f"temperature unit {raw_unit!r} not convertible to K",
            "location": loc,
        })
        return None, "K", diags

    table = _LINEAR.get(kind)
    key = unit.lower()
    if table is not None and key in table:
        return value * table[key], CANONICAL_UNIT[kind], diags
    diags.append({
        "severity": "error", "code": "UNIT_UNCONVERTIBLE",
        "message": f"unit {raw_unit!r} not convertible to {CANONICAL_UNIT.get(kind)}",
        "location": loc,
    })
    return None, CANONICAL_UNIT.get(kind), diags


def normalize_series(raw_values, raw_unit, kind: str, location=None):
    """Normalize a list of values; per-element diagnostics carry the index."""
    out, diags = [], []
    for i, v in enumerate(raw_values):
        loc = dict(location or {})
        loc["index"] = i
        val, unit, ds = normalize_quantity(v, raw_unit, kind, loc)
        out.append(val)
        diags.extend(ds)
    return out, CANONICAL_UNIT.get(kind), diags
