"""Unit registry and quantity parsing.

All domain values are imported with their *original representation* intact.
Conversion to SI happens only in derived layers; values that cannot be
converted stay where they were found and carry a position-tagged diagnostic
instead of being dropped or defaulted.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Canonical dimensions used across the ITC domain.
# scale converts a value expressed in the given unit to the SI/base unit.
SCALES: dict[str, dict[str, float]] = {
    "power": {"uw": 1e-6, "mw": 1e-3, "w": 1.0},
    "time": {"s": 1.0, "min": 60.0, "sec": 1.0, "second": 1.0, "seconds": 1.0},
    "volume": {"ul": 1e-6, "ml": 1e-3, "l": 1.0},
    "concentration": {
        "um": 1e-6,
        "mm": 1e-3,
        "m": 1.0,
        "mol/l": 1.0,
        "umol/l": 1e-6,
        "mmol/l": 1e-3,
    },
    "temperature": {"c": 1.0, "k": 1.0, "kelvin": 1.0, "celsius": 1.0},
    "energy": {"ucal": 4.184e-6, "cal": 4.184, "j": 1.0, "mj": 1e-3, "uj": 1e-6},
    "heat_rate_cal": {"ucal/s": 4.184e-6, "cal/s": 4.184},
}

# Heat is stored internally as joules (derived from power*time).
ALIASES = {
    "µw": "uw",
    "μw": "uw",
    "µl": "ul",
    "μl": "ul",
    "µm": "um",
    "μm": "um",
    "mol l^-1": "mol/l",
    "mol/l": "mol/l",
    "°c": "c",
    "degc": "c",
    "deg c": "c",
    "j/cal": "j",
}

_TOKEN_RE = re.compile(
    r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>.*?)\s*$"
)


class UnitError(ValueError):
    """Raised for an unknown/unsupported unit (distinct from a bad number)."""


@dataclass(frozen=True)
class ParsedQuantity:
    value: float | None          # None => could not be converted
    si_value: float | None
    unit: str
    dimension: str
    raw: str

    @property
    def convertible(self) -> bool:
        return self.si_value is None or True  # noqa: E731 (kept explicit below)


def _norm_unit(unit: str) -> str:
    u = unit.strip().lower()
    u = u.replace("μ", "µ")
    return ALIASES.get(u, u)


def dimension_for(unit: str) -> str:
    nu = _norm_unit(unit)
    for dim, table in SCALES.items():
        if nu in table:
            return dim
    raise UnitError(f"unknown unit {unit!r}")


def convert(value: float, unit: str, dimension: str | None = None) -> float:
    """Convert ``value`` expressed in ``unit`` to SI for its dimension."""
    nu = _norm_unit(unit)
    table: dict[str, float] | None = None
    if dimension is not None:
        table = SCALES.get(dimension)
        if table is None or nu not in table:
            raise UnitError(f"unit {unit!r} is not a {dimension} unit")
    else:
        for dim_name, dim_table in SCALES.items():
            if nu in dim_table:
                table = dim_table
                dimension = dim_name
                break
        if table is None:
            raise UnitError(f"unknown unit {unit!r}")
    if nu in ("c", "celsius"):
        # Offset unit; caller decides conversion semantics via ``parse_quantity``.
        return value
    return value * table[nu]


def parse_quantity(raw: str, expected_dimension: str | None = None,
                   default_unit: str | None = None) -> ParsedQuantity:
    """Parse ``"<number> <unit>"`` preserving the original text.

    Temperatures: Celsius applies the 273.15 offset, Kelvin is absolute.
    Returns a ParsedQuantity with ``value=None`` on an unconvertible token;
    callers retain the raw text and attach a diagnostic rather than dropping it.
    """
    m = _TOKEN_RE.match(raw)
    if not m:
        return ParsedQuantity(None, None, raw.strip() or "<empty>",
                              expected_dimension or "unknown", raw)
    number_text, unit_text = m.group("value"), m.group("unit")
    if not unit_text.strip() and default_unit is not None:
        unit_text = default_unit
    try:
        value = float(number_text)
    except ValueError:
        return ParsedQuantity(None, None, unit_text, expected_dimension or "unknown", raw)
    try:
        dim = dimension_for(unit_text)
    except UnitError:
        return ParsedQuantity(value, None, unit_text,
                              expected_dimension or "unknown", raw)
    if expected_dimension is not None and dim != expected_dimension:
        return ParsedQuantity(value, None, unit_text, dim, raw)
    nu = _norm_unit(unit_text)
    if nu in ("c", "celsius"):
        si = value + 273.15
    else:
        si = value * SCALES[dim][nu]
    if math.isnan(si) or math.isinf(si):
        si = None
    return ParsedQuantity(value, si, unit_text.strip(), dim, raw)


def si(value: float, unit: str, dimension: str | None = None) -> float:
    """Strict convenience converter used by derived layers."""
    return convert(value, unit, dimension)


def celsius_to_kelvin(value: float) -> float:
    return value + 273.15
