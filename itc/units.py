"""Unit / coordinate-base handling.

The import layer KEEPS THE ORIGINAL REPRESENTATION (raw string + declared
unit).  Normalisation to the canonical internal bases happens only here in the
derived layer.  Values that cannot be converted are returned as ``None`` with a
diagnostic and stay in their original location - they are never dropped to make
plotting possible.

Canonical internal bases:
    time          s (epoch anchor = material ``time_zero`` ISO string)
    power         ucal/s
    temperature   K
    volume        L
    concentration M (mol/L)
    energy        cal
    heat rate per injection: cal
"""
from __future__ import annotations

from typing import Any

from .errors import Diagnostic

# multiplicative factors to canonical
_FACTORS: dict[tuple[str, str], float] = {
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "second"): 1.0,
    ("time", "seconds"): 1.0,
    ("time", "ms"): 1e-3,
    ("time", "millisecond"): 1e-3,
    ("time", "min"): 60.0,
    ("time", "minute"): 60.0,
    ("time", "minutes"): 60.0,
    ("power", "ucal/s"): 1.0,
    ("power", "cal/s"): 1e6,
    ("power", "mcal/s"): 1e3,
    ("power", "uW"): 0.239005736,   # 1 uW = 0.239006 ucal/s
    ("power", "uw"): 0.239005736,
    ("power", "mw"): 239.005736,
    ("energy", "cal"): 1.0,
    ("energy", "kcal"): 1e3,
    ("energy", "ucal"): 1e-6,
    ("energy", "mcal"): 1e-3,
    ("energy", "j"): 0.239005736,
    ("energy", "joule"): 0.239005736,
    ("volume", "l"): 1.0,
    ("volume", "ml"): 1e-3,
    ("volume", "ul"): 1e-6,
    ("volume", "uL"): 1e-6,
    ("concentration", "m"): 1.0,
    ("concentration", "mm"): 1e-3,
    ("concentration", "um"): 1e-6,
    ("concentration", "molar"): 1.0,
}

_TEMP_K: dict[str, str] = {"k", "kelvin"}
_TEMP_C: dict[str, str] = {"c", "celsius", "degc", "deg c", "°c"}


def normalize_value(dimension: str, raw_value: Any, unit: str | None,
                    path: str, at: dict[str, Any] | None = None
                    ) -> tuple[float | None, Diagnostic | None]:
    """Return (canonical value, optional diagnostic).

    A missing/blank value is a non-fatal UNCONVERTIBLE_VALUE diagnostic.
    An unknown unit is UNKNOWN_UNIT.  The caller retains the raw pair.
    """
    at = at or {}
    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        return None, Diagnostic(
            "UNCONVERTIBLE_VALUE", path,
            "value is blank; original slot retained, not used in derivation",
            raw=raw_value, at=at)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None, Diagnostic(
            "UNCONVERTIBLE_VALUE", path,
            f"cannot parse {raw_value!r} as {dimension}; kept in place",
            raw=raw_value, at=at)
    u = (unit or "").strip()
    ul = u.lower()
    if dimension == "temperature":
        if ul in _TEMP_K:
            return value, None
        if ul in _TEMP_C:
            return value + 273.15, None
        return None, Diagnostic(
            "UNKNOWN_UNIT", path,
            f"unknown temperature unit {u!r}; value retained raw",
            raw=raw_value, at={**at, "unit": u})
    factor = _FACTORS.get((dimension, u))
    if factor is None:
        factor = _FACTORS.get((dimension, ul))
    if factor is None:
        return None, Diagnostic(
            "UNKNOWN_UNIT", path,
            f"unknown {dimension} unit {u!r}; value retained raw",
            raw=raw_value, at={**at, "unit": u})
    return value * factor, None
