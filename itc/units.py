"""Units, raw-token parsing and normalisation.

Design rule from the case contract: the raw representation is preserved at
import time; only the derived layer normalises.  Values that cannot be
converted stay at their original location with a diagnostic.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .errors import FormatCorrupt, SemanticUnsolvable

# dimension -> canonical unit token
CANONICAL = {
    "time": "s",
    "power": "W",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
    "energy": "J",
    "enthalpy": "J/mol",
    "heat": "cal",        # per-injection heat is conventionally reported in cal/ucal
}

# factor: value_in_canonical = value_in_declared * factor (temperature excepted)
_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
             "ms": 1e-3, "min": 60.0, "minute": 60.0, "minutes": 60.0,
             "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0},
    "power": {"W": 1.0, "w": 1.0, "watt": 1.0, "watts": 1.0,
              "mW": 1e-3, "mw": 1e-3,
              "uW": 1e-6, "uw": 1e-6, "µW": 1e-6, "μw": 1e-6,
              "nW": 1e-9, "nw": 1e-9},
    "volume": {"L": 1.0, "l": 1.0,
               "mL": 1e-3, "ml": 1e-3,
               "uL": 1e-6, "ul": 1e-6, "µL": 1e-6, "μl": 1e-6,
               "nL": 1e-9, "nl": 1e-9},
    "concentration": {"M": 1.0, "molar": 1.0,
                      "mM": 1e-3, "mm": 1e-3,
                      "uM": 1e-6, "um": 1e-6, "µM": 1e-6, "μm": 1e-6,
                      "nM": 1e-9, "nm": 1e-9,
                      "pM": 1e-12, "pm": 1e-12},
    "energy": {"J": 1.0, "j": 1.0, "mJ": 1e-3, "mj": 1e-3,
               "cal": 4.184, "Cal": 4.184},
    "heat": {"cal": 1.0, "mcal": 1e-3, "ucal": 1e-6, "µcal": 1e-6,
             "J": 4.184, "mJ": 4.184e-3, "uJ": 4.184e-6},
    "enthalpy": {"cal/mol": 4.184, "kcal/mol": 4184.0,
                 "J/mol": 1.0, "j/mol": 1.0,
                 "kJ/mol": 1000.0, "kj/mol": 1000.0,
                 "cal/mmol": 4.184, "J/mmol": 1.0, "kJ/mmol": 1000.0},
}

_TEMP_UNITS = {"K", "kelvin", "C": None, "degC": None, "°C": None,
               "F": None, "degF": None, "°F": None}

_NUMBER_RE = re.compile(
    r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def normalise_unit(dimension: str, unit: str) -> str:
    """Return canonical token for a declared unit, or raise FormatCorrupt."""
    if dimension == "temperature":
        u = unit.strip()
        if u in ("K", "kelvin", "Kelvin"):
            return "K"
        if u in ("C", "degC", "°C", "celsius", "Celsius"):
            return "degC"
        if u in ("F", "degF", "°F", "fahrenheit", "Fahrenheit"):
            return "degF"
        raise FormatCorrupt(f"unknown temperature unit {unit!r}", location="header")
    table = _FACTORS.get(dimension)
    if table is None:
        raise FormatCorrupt(f"unknown dimension {dimension!r}")
    if unit.strip() not in table:
        raise FormatCorrupt(
            f"unknown {dimension} unit {unit!r}", location="header")
    return CANONICAL[dimension]


def convert(dimension: str, value: float, unit: str) -> float:
    """Convert a numeric value from a declared unit to the canonical unit."""
    if dimension == "temperature":
        canon = normalise_unit("temperature", unit)
        if canon == "K":
            return value
        if canon == "degC":
            return value + 273.15
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    table = _FACTORS[dimension]
    if unit.strip() not in table:
        raise FormatCorrupt(f"unknown {dimension} unit {unit!r}", location="header")
    return value * table[unit.strip()]


def parse_number(token: str, location: str = "") -> float:
    """Parse a number; raw representation lives with the caller."""
    t = token.strip()
    if not t:
        raise FormatCorrupt("empty numeric value", location=location, raw=token)
    # tolerate a single embedded unit only when separated by whitespace; the
    # raw token is still retained upstream.
    if not _NUMBER_RE.match(t):
        raise FormatCorrupt(f"not a number: {token!r}", location=location, raw=token)
    try:
        return float(t)
    except ValueError as exc:  # pragma: no cover - regex covers this
        raise FormatCorrupt(str(exc), location=location, raw=token)


def parse_quantity(token: str, dimension: str, location: str = ""):
    """Parse ``"<number> <unit>"`` (or number with separately declared unit).

    Returns ``(canonical_value, declared_unit_or_None, canonical_unit_or_None)``.
    A value with no known unit returns ``(None, declared, None)`` and the
    *caller* attaches a SEMANTIC diagnostic while keeping the raw token in
    place.
    """
    t = token.strip()
    parts = t.rsplit(" ", 1)
    number_part = t
    unit_part: Optional[str] = None
    if len(parts) == 2 and not _NUMBER_RE.match(parts[1].strip()):
        number_part, unit_part = parts[0].strip(), parts[1].strip()
    value = parse_number(number_part, location=location)
    if unit_part is None:
        return value, None, CANONICAL.get(dimension) if dimension in (
            "time", "power", "volume", "concentration", "energy", "heat",
            "enthalpy") else None
    try:
        canon = normalise_unit(dimension, unit_part)
    except FormatCorrupt:
        return None, unit_part, None
    if dimension == "temperature":
        return convert("temperature", value, unit_part), unit_part, canon
    table = _FACTORS[dimension]
    return value * table[unit_part], unit_part, canon


def finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
