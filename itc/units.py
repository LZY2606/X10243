"""Unit / coordinate-basis handling.

Raw material payloads always keep their original representation.  The derived
layer calls :func:`normalize`; values that cannot be converted stay in place
and produce a positioned diagnostic instead of being dropped or defaulted.
"""
from __future__ import annotations

_SCALE = {
    ("uL", "L"): 1e-6, ("mL", "L"): 1e-3, ("L", "L"): 1.0,
    ("uL", "mL"): 1e-3, ("mL", "mL"): 1.0,
    ("nM", "M"): 1e-9, ("uM", "M"): 1e-6, ("mM", "M"): 1e-3, ("M", "M"): 1.0,
    ("s", "s"): 1.0, ("min", "s"): 60.0,
    ("uJ/s", "uJ/s"): 1.0, ("ucal/s", "uJ/s"): 4.184, ("mcal/s", "uJ/s"): 4184.0,
    ("uJ", "uJ"): 1.0, ("ucal", "uJ"): 4.184, ("cal", "uJ"): 4.184e6,
    ("J", "J"): 1.0, ("uJ", "J"): 1e-6, ("J", "uJ"): 1e6,
    ("kJ/mol", "J/mol"): 1e3, ("J/mol", "J/mol"): 1.0, ("kcal/mol", "J/mol"): 4184.0,
    ("K", "K"): 1.0,
}


def convert(value, from_unit, to_unit):
    """Return converted value, or None when the conversion is impossible."""
    if from_unit == to_unit:
        return value
    factor = _SCALE.get((from_unit, to_unit))
    if factor is not None:
        return value * factor
    if from_unit == "degC" and to_unit == "K":
        return value + 273.15
    return None


def normalize(quantity, target_unit, material_kind, path):
    """Normalize a ``{"value", "unit"}`` blob.

    Returns ``(value, diagnostics)``; value is None when conversion fails and
    the original representation is intentionally left to the caller (material
    payload) with a positioned diagnostic.
    """
    diags = []
    if not isinstance(quantity, dict) or "value" not in quantity or "unit" not in quantity:
        return None, [{
            "code": "quantity_malformed",
            "message": f"expected {{value, unit}} at {path}",
            "position": {"material": material_kind, "path": path},
            "severity": "error",
        }]
    raw = quantity["value"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, [{
            "code": "quantity_not_numeric",
            "message": f"raw value {raw!r} at {path} is not numeric; kept in original representation",
            "position": {"material": material_kind, "path": path, "raw": raw},
            "severity": "error",
        }]
    out = convert(value, quantity["unit"], target_unit)
    if out is None:
        return None, [{
            "code": "unit_unconvertible",
            "message": f"cannot convert {quantity['unit']!r} to {target_unit!r} at {path}; raw value retained",
            "position": {"material": material_kind, "path": path,
                         "raw_value": raw, "raw_unit": quantity["unit"]},
            "severity": "error",
        }]
    return out, diags


def as_display(quantity):
    if isinstance(quantity, dict) and "value" in quantity and "unit" in quantity:
        return f"{quantity['value']} {quantity['unit']}"
    return str(quantity)
