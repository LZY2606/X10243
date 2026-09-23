"""Quantities, units and coordinate bases.

Rule (from the brief): the *original* representation is preserved on import;
normalisation happens only in the derived layer. A value that cannot be
converted stays where it is with a diagnostic instead of being dropped or
silently defaulted.

Every derived numeric carries a unit and a coordinate/time basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .errors import diagnostic

# Canonical internal units ---------------------------------------------------
# concentration : M (mol/L)
# volume        : L
# time          : s
# power         : µcal/s
# heat/energy   : cal  (power is µcal/s; integration yields µcal -> we keep cal)
# temperature   : K

# conversion factor -> canonical unit, keyed by (kind, unit-token lower-case)
_FACTORS: dict[tuple[str, str], float] = {
    ("concentration", "m"): 1.0,
    ("concentration", "mm"): 1e-3,
    ("concentration", "um"): 1e-6,
    ("concentration", "µm"): 1e-6,
    ("concentration", "nm"): 1e-9,
    ("concentration", "mmolar"): 1e-3,
    ("concentration", "umolar"): 1e-6,
    ("volume", "l"): 1.0,
    ("volume", "ml"): 1e-3,
    ("volume", "µl"): 1e-6,
    ("volume", "ul"): 1e-6,
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "seconds"): 1.0,
    ("time", "min"): 60.0,
    ("time", "minute"): 60.0,
    ("time", "minutes"): 60.0,
    ("power", "µcal/s"): 1.0,
    ("power", "ucal/s"): 1.0,
    ("power", "cal/s"): 1e6,
    ("power", "mcal/s"): 1e3,
    ("power", "µw"): 0.239005736,
    ("power", "uw"): 0.239005736,
    ("energy", "cal"): 1.0,
    ("energy", "kcal"): 1e3,
    ("energy", "µcal"): 1e-6,
    ("energy", "ucal"): 1e-6,
    ("energy", "mcal"): 1e-3,
    ("energy", "j"): 0.239005736,
    ("energy", "mj"): 0.239005736e-3,
}

CANONICAL_UNIT = {
    "concentration": "M",
    "volume": "L",
    "time": "s",
    "power": "µcal/s",
    "energy": "cal",
    "temperature": "K",
}


@dataclass(frozen=True)
class Basis:
    """Coordinate / time basis so every plotted number knows its frame."""

    time_zero: str = "series_first_sample"  # origin of the t axis
    time_unit: str = "s"
    energy_unit: str = "cal"
    power_unit: str = "µcal/s"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_zero": self.time_zero,
            "time_unit": self.time_unit,
            "energy_unit": self.energy_unit,
            "power_unit": self.power_unit,
            "note": self.note,
        }


def is_known_unit(kind: str, unit: str) -> bool:
    if kind == "temperature":
        return unit.strip().lower() in {"k", "c", "°c", "degc"}
    return (kind, unit.strip().lower()) in _FACTORS


def convert(kind: str, value: float, unit: str) -> float:
    """Convert ``value`` expressed in ``unit`` to the canonical unit of ``kind``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"numeric required, got {type(value)!r}")
    u = unit.strip()
    if kind == "temperature":
        ul = u.lower()
        if ul == "k":
            return float(value)
        if ul in {"c", "°c", "degc"}:
            return float(value) + 273.15
        raise KeyError(f"unknown temperature unit {unit!r}")
    return float(value) * _FACTORS[(kind, u.lower())]


def to_canonical(kind: str, value_canonical: float, unit: str | None = None) -> float:
    """Identity/escape hatch: value is already canonical (used by derived layer)."""
    return float(value_canonical)


def parse_quantity(raw: Any, kind: str, *, location: str) -> dict[str, Any]:
    """Parse a raw cell into ``{raw, value, unit, basis, diagnostics}``.

    Accepts a number (unit then required by caller context) or a string such as
    ``"2.5 µL"``. An unconvertible value keeps ``raw`` and a diagnostic;
    ``value`` is None but the entry is never dropped.
    """
    diag: list[dict[str, Any]] = []
    unit: str | None = None
    numeric: float | None = None

    if isinstance(raw, bool):  # bool is an int subclass; reject explicitly
        diag.append(diagnostic("not_a_quantity", f"{raw!r} is not a quantity", location=location))
        return {"raw": raw, "value": None, "unit": unit, "diagnostics": diag}

    if isinstance(raw, (int, float)):
        numeric = float(raw)
        return {"raw": raw, "value": numeric, "unit": None, "diagnostics": diag,
                "as_imported_number": True}

    if raw is None:
        diag.append(diagnostic("missing_value", "empty cell; no default substituted",
                               location=location))
        return {"raw": raw, "value": None, "unit": unit, "diagnostics": diag}

    text = str(raw).strip()
    if text == "":
        diag.append(diagnostic("missing_value", "empty cell; no default substituted",
                               location=location))
        return {"raw": raw, "value": None, "unit": unit, "diagnostics": diag}

    token, num_text = _split_value_unit(text)
    try:
        numeric = float(num_text)
    except ValueError:
        diag.append(diagnostic("not_a_number", f"cannot read number from {text!r}",
                               location=location))
        return {"raw": raw, "value": None, "unit": token or None, "diagnostics": diag}

    if token is None:
        # dimensionless import: keep number but flag that the unit must be provided
        diag.append(diagnostic("unit_missing",
                               "number imported without an explicit unit; "
                               "canonicalisation deferred until a unit is attached",
                               location=location))
        return {"raw": raw, "value": numeric, "unit": None, "diagnostics": diag}

    if not is_known_unit(kind, token):
        diag.append(diagnostic("unknown_unit", f"unit {token!r} is not known for {kind}",
                               location=location))
        return {"raw": raw, "value": numeric, "unit": token, "diagnostics": diag}

    try:
        converted = convert(kind, numeric, token)
    except (KeyError, ValueError):
        diag.append(diagnostic("unconvertible", f"cannot convert {text!r} to {kind}",
                               location=location))
        return {"raw": raw, "value": numeric, "unit": token, "diagnostics": diag}

    return {"raw": raw, "value": converted, "unit": CANONICAL_UNIT[kind],
            "original_unit": token, "diagnostics": diag}


def _split_value_unit(text: str) -> tuple[str | None, str]:
    """Split ``"2.5 µL"`` into unit token ``"µL"`` and numeric text ``"2.5"``."""
    i = 0
    while i < len(text) and (text[i].isdigit() or text[i] in "+-.eE " ):
        i += 1
    num = text[:i].strip()
    unit = text[i:].strip()
    # scientific notation with trailing spaces e.g. "1.2 e-3" -> already fine
    return (unit or None), num
