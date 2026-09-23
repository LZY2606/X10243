"""Unit handling.

Raw representations are preserved verbatim inside material content. Only the
derived (compute) layer normalizes to canonical units. Values that cannot be
converted stay in place and produce a located diagnostic; nothing is dropped
to make plotting easier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .results import Diagnostic

# dimension -> (canonical unit, {unit: (multiplicative factor, additive offset)})
_TABLE = {
    "time": ("s", {"s": (1.0, 0.0), "sec": (1.0, 0.0), "min": (60.0, 0.0)}),
    "power": ("uJ/s", {
        "uJ/s": (1.0, 0.0), "µJ/s": (1.0, 0.0),
        "ucal/s": (4.184, 0.0), "µcal/s": (4.184, 0.0), "mcal/s": (4184.0, 0.0),
        "nJ/s": (1e-3, 0.0),
    }),
    "power_rate": ("uJ/s^2", {
        "uJ/s^2": (1.0, 0.0), "µJ/s^2": (1.0, 0.0),
        "ucal/s^2": (4.184, 0.0), "µcal/s^2": (4.184, 0.0),
    }),
    "volume": ("uL", {"uL": (1.0, 0.0), "µL": (1.0, 0.0), "mL": (1000.0, 0.0), "nL": (1e-3, 0.0)}),
    "concentration": ("uM", {
        "uM": (1.0, 0.0), "µM": (1.0, 0.0),
        "mM": (1000.0, 0.0), "M": (1e6, 0.0), "nM": (1e-3, 0.0),
    }),
    "temperature": ("K", {"K": (1.0, 0.0), "°C": (1.0, 273.15), "C": (1.0, 273.15)}),
    "cell_volume": ("mL", {"mL": (1.0, 0.0), "uL": (1e-3, 0.0), "µL": (1e-3, 0.0)}),
    "heat": ("uJ", {"uJ": (1.0, 0.0), "µJ": (1.0, 0.0), "ucal": (4.184, 0.0), "mJ": (1000.0, 0.0)}),
    "molar_enthalpy": ("uJ/mol", {
        "uJ/mol": (1.0, 0.0), "µJ/mol": (1.0, 0.0),
        "kJ/mol": (1e9, 0.0), "kcal/mol": (4.184e9, 0.0),
    }),
}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_QRE = re.compile(rf"^\s*({_NUM})\s*(.+?)\s*$")


@dataclass
class Quantity:
    raw: str
    value: float
    unit: str


def canonical_unit(dimension: str) -> str:
    return _TABLE[dimension][0]


def parse_quantity(obj) -> Optional[Quantity]:
    """Accept '120 µM', {'value': 120, 'unit': 'µM'}. None if unparseable."""
    if isinstance(obj, dict):
        try:
            value = float(obj["value"])
        except (KeyError, TypeError, ValueError):
            return None
        unit = str(obj.get("unit", "")).strip()
        return Quantity(raw=f"{obj.get('value')} {unit}".strip(), value=value, unit=unit)
    if isinstance(obj, str):
        m = _QRE.match(obj)
        if not m:
            return None
        return Quantity(raw=obj, value=float(m.group(1)), unit=m.group(2))
    return None


def normalize(quantity: Quantity, dimension: str, position: str) -> Tuple[Optional[float], Optional[Diagnostic]]:
    """Convert to the canonical unit of the dimension. Unconvertible values
    return (None, diagnostic) and stay raw in the material content."""
    if dimension not in _TABLE:
        return None, Diagnostic("unknown_dimension", f"no unit table for dimension '{dimension}'", position)
    _, table = _TABLE[dimension]
    entry = table.get(quantity.unit)
    if entry is None:
        return None, Diagnostic(
            "unit_unconvertible",
            f"unit '{quantity.unit}' cannot be converted to {_TABLE[dimension][0]}; raw value kept in place",
            position,
        )
    factor, offset = entry
    return quantity.value * factor + offset, None


def normalize_quantity(obj, dimension: str, position: str):
    """Full pipeline: parse + normalize. Returns (value, diagnostic)."""
    q = parse_quantity(obj)
    if q is None:
        return None, Diagnostic("quantity_unparseable", f"cannot parse quantity from {obj!r}", position)
    return normalize(q, dimension, position)
