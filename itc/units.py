"""Unit handling.

Raw materials keep their original representation. Normalisation to canonical
units happens only in the derived layer. Values that cannot be converted stay
in place (raw) and carry a diagnostic; they are never silently dropped or
replaced with defaults.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .diagnostics import Diagnostic

# dimension -> {unit: (scale_to_canonical, offset)} ; canonical unit is table key with scale 1 and offset 0
_TABLE = {
    "time": ({"s": (1.0, 0.0), "sec": (1.0, 0.0), "min": (60.0, 0.0)}, "s"),
    "power": ({"uJ/s": (1.0, 0.0), "uW": (1.0, 0.0), "uj/s": (1.0, 0.0)}, "uJ/s"),
    "volume": ({"uL": (1.0, 0.0), "µL": (1.0, 0.0), "mL": (1000.0, 0.0)}, "uL"),
    "concentration": ({"uM": (1.0, 0.0), "mM": (1000.0, 0.0), "M": (1.0e6, 0.0)}, "uM"),
    "temperature": ({"K": (1.0, 0.0), "degC": (1.0, 273.15)}, "K"),
    "energy": ({"uJ": (1.0, 0.0), "mJ": (1000.0, 0.0)}, "uJ"),
    "molar_energy": ({"kJ/mol": (1.0, 0.0), "kcal/mol": (4.184, 0.0)}, "kJ/mol"),
}


def canonical_unit(dimension: str) -> str:
    return _TABLE[dimension][1]


def conversion(unit: str, dimension: str):
    """Return (scale, offset) to canonical units, or None if unconvertible."""
    table, _ = _TABLE[dimension]
    return table.get(unit)


def normalize(quantity, dimension: str, path: str = "") -> Tuple[Optional[float], Optional[str], Optional[Diagnostic]]:
    """Return (canonical_value, canonical_unit, diagnostic).

    On success diagnostic is None. On failure value/unit are None and the
    diagnostic explains why; callers must keep the raw quantity in place.
    """
    if not isinstance(quantity, dict) or "value" not in quantity or "unit" not in quantity:
        return None, None, Diagnostic(
            code="quantity_malformed",
            message=f"expected {{value, unit}} for dimension {dimension}",
            path=path,
        )
    table, canon = _TABLE[dimension]
    unit = quantity["unit"]
    if unit not in table:
        return None, None, Diagnostic(
            code="unit_unconvertible",
            message=f"unit {unit!r} cannot be converted to {canon} (dimension {dimension}); raw value kept in place",
            path=path,
        )
    scale, offset = table[unit]
    try:
        value = float(quantity["value"]) * scale + offset
    except (TypeError, ValueError):
        return None, None, Diagnostic(
            code="quantity_not_numeric",
            message=f"value {quantity['value']!r} is not numeric",
            path=path,
        )
    return value, canon, None
