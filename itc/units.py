"""Unit handling.

Raw representations are always preserved by callers (store layer keeps the
raw JSON). This module only produces the *derived* normalized view. Values
that cannot be converted stay where they were and produce a diagnostic;
nothing is silently dropped or defaulted.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .diagnostics import Diagnostic, SEMANTIC_UNSOLVABLE

CANONICAL = {
    "time": "s",
    "power": "uW",
    "energy": "uJ",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
    "enthalpy": "uJ/umol",
}

# factor tables map an alias -> (multiplier, offset) to reach canonical unit
_TABLES = {
    "time": {"s": (1.0, 0.0), "sec": (1.0, 0.0), "min": (60.0, 0.0)},
    "power": {
        "uW": (1.0, 0.0), "mW": (1.0e3, 0.0),
        "ucal/s": (4.184, 0.0), "mcal/s": (4.184e3, 0.0),
    },
    "energy": {"uJ": (1.0, 0.0), "mJ": (1.0e3, 0.0), "ucal": (4.184, 0.0)},
    "volume": {"uL": (1.0, 0.0), "mL": (1.0e3, 0.0), "nL": (1.0e-3, 0.0),
               "L": (1.0e6, 0.0)},
    "concentration": {"uM": (1.0, 0.0), "mM": (1.0e3, 0.0),
                      "nM": (1.0e-3, 0.0), "M": (1.0e6, 0.0)},
    "temperature": {"K": (1.0, 0.0), "C": (1.0, 273.15)},
    "enthalpy": {"uJ/umol": (1.0, 0.0), "kJ/mol": (1.0e3, 0.0),
                 "kcal/mol": (4184.0, 0.0), "J/mol": (1.0, 0.0),
                 "cal/mol": (4.184, 0.0)},
}


def _alias(unit: str) -> str:
    u = unit.strip()
    u = u.replace("µ", "u").replace("μ", "u")  # micro sign / greek mu
    u = u.replace("°C", "C").replace("degC", "C")
    u = u.replace("μL", "uL").replace("ul", "uL") if u == "ul" else u
    return u


def convert(kind: str, value: float, unit: str,
            location: str = "") -> Tuple[Optional[float], Optional[Diagnostic]]:
    """Convert `value` expressed in `unit` to the canonical unit of `kind`.
    Returns (canonical_value, None) or (None, Diagnostic)."""
    if kind not in _TABLES:
        return None, Diagnostic("UNIT_KIND_UNKNOWN", SEMANTIC_UNSOLVABLE,
                                location, f"unknown quantity kind '{kind}'")
    table = _TABLES[kind]
    alias = _alias(unit)
    if alias not in table:
        return None, Diagnostic(
            "UNIT_UNCONVERTIBLE", SEMANTIC_UNSOLVABLE, location,
            f"cannot convert unit '{unit}' to canonical "
            f"'{CANONICAL[kind]}' for kind '{kind}'")
    factor, offset = table[alias]
    return value * factor + offset, None


def canonical_unit(kind: str) -> str:
    return CANONICAL[kind]
