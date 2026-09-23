"""Unit handling.

Raw imported values are preserved verbatim in the material payloads. The
derived layer normalizes to canonical units through this module. Values whose
unit cannot be converted stay in place (the raw payload is untouched) and a
positioned diagnostic is emitted; nothing is dropped for plotting convenience.

Canonical units:
  time          -> s           (coordinate basis: seconds since trace start)
  power         -> uJ/s        (1 ucal/s = 4.184 uJ/s)
  heat          -> uJ
  volume        -> uL
  concentration -> uM
  temperature   -> K
"""
from __future__ import annotations

from typing import Optional, Tuple

from .diagnostics import Diagnostic

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "heat": "uJ",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
}

_FACTORS = {
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0},
    "power": {"uJ/s": 1.0, "uJ.s-1": 1.0, "ucal/s": 4.184, "mcal/s": 4184.0},
    "heat": {"uJ": 1.0, "ucal": 4.184, "mJ": 1000.0, "cal": 4184.0},
    "volume": {"uL": 1.0, "mL": 1000.0, "L": 1.0e6},
    "concentration": {"uM": 1.0, "mM": 1000.0, "M": 1.0e6, "nM": 1.0e-3},
}


def normalize(value: float, unit: str, quantity: str, path: str = "") -> Tuple[Optional[float], Optional[Diagnostic]]:
    """Convert `value` expressed in `unit` to the canonical unit of `quantity`.

    Returns (canonical_value, None) or (None, Diagnostic) when the unit is
    unknown for that quantity.
    """
    if quantity == "temperature":
        if unit in ("K", "kelvin"):
            return float(value), None
        if unit in ("C", "degC", "°C"):
            return float(value) + 273.15, None
        return None, Diagnostic(
            code="unit_unknown",
            message=f"cannot convert temperature unit {unit!r} to K",
            path=path,
        )
    table = _FACTORS.get(quantity)
    if table is None:
        return None, Diagnostic(code="quantity_unknown", message=f"unknown quantity {quantity!r}", path=path)
    factor = table.get(unit)
    if factor is None:
        return None, Diagnostic(
            code="unit_unknown",
            message=f"cannot convert {quantity} unit {unit!r} to {CANONICAL[quantity]}",
            path=path,
        )
    return float(value) * factor, None


def denormalize(value: float, quantity: str, unit: str) -> float:
    """Canonical -> display unit (used by the UI layer only)."""
    if quantity == "temperature":
        return value - 273.15 if unit in ("C", "degC", "°C") else value
    factor = _FACTORS[quantity][unit]
    return value / factor
