"""Quantity handling.

Import keeps the *raw representation* (the exact token plus its declared unit).
Normalisation to internal SI-ish canonical units only happens in the derived
layer through :class:`Quantity`.  Values that cannot be converted stay in place
and carry a structured diagnostic instead of being dropped or defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class UnitError(ValueError):
    """Raised when a (value, unit) pair cannot be converted."""


# Linear factors: unit -> (factor, canonical_unit).  value_canon = value * factor
LINEAR_UNITS: dict[str, dict[str, tuple[float, str]]] = {
    "time": {
        "s": (1.0, "s"),
        "sec": (1.0, "s"),
        "second": (1.0, "s"),
        "seconds": (1.0, "s"),
        "min": (60.0, "s"),
        "minute": (60.0, "s"),
        "minutes": (60.0, "s"),
    },
    "power": {
        "W": (1.0, "W"),
        "mW": (1e-3, "W"),
        "uW": (1e-6, "W"),
        "µW": (1e-6, "W"),
        "µcal/s": (4.184e-6, "W"),
        "ucal/s": (4.184e-6, "W"),
        "cal/s": (4.184, "W"),
    },
    "energy": {
        "J": (1.0, "J"),
        "mJ": (1e-3, "J"),
        "uJ": (1e-6, "J"),
        "µJ": (1e-6, "J"),
        "cal": (4.184, "J"),
        "cal/mol": (4.184, "J/mol"),
        "kcal/mol": (4184.0, "J/mol"),
        "kJ/mol": (1000.0, "J/mol"),
        "J/mol": (1.0, "J/mol"),
    },
    "volume": {
        "L": (1.0, "L"),
        "mL": (1e-3, "L"),
        "uL": (1e-6, "L"),
        "µL": (1e-6, "L"),
        "liter": (1.0, "L"),
    },
    "concentration": {
        "M": (1.0, "M"),
        "mM": (1e-3, "M"),
        "uM": (1e-6, "M"),
        "µM": (1e-6, "M"),
        "nM": (1e-9, "M"),
        "mol/L": (1.0, "M"),
    },
    "temperature_difference": {
        "K": (1.0, "K"),
        "C delta": (1.0, "K"),
    },
}

# Preferred reporting units (what the UI exposes as the canonical basis).
CANONICAL_UNIT = {
    "time": "s",
    "power": "uW",
    "energy": "uJ",
    "molar_energy": "kJ/mol",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
}


@dataclass(frozen=True)
class Quantity:
    """A scalar measurement that remembers its raw representation.

    ``raw_text`` is the exact token found in the source file; ``raw_unit`` is
    the unit declared next to it.  ``value`` is the normalised numeric value in
    ``canonical_unit``; it is ``None`` when conversion was impossible, in which
    case ``diagnostic`` explains why (the raw token is never dropped).
    """

    dimension: str
    raw_text: str
    raw_unit: str | None
    value: float | None
    canonical_unit: str | None
    diagnostic: str | None = None

    @property
    def convertible(self) -> bool:
        return self.value is not None and self.diagnostic is None

    def require(self) -> float:
        """Return the normalised value or raise :class:`UnitError`."""
        if not self.convertible:
            raise UnitError(self.diagnostic or "unconvertible quantity")
        return self.value  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "raw_text": self.raw_text,
            "raw_unit": self.raw_unit,
            "value": self.value,
            "canonical_unit": self.canonical_unit,
            "diagnostic": self.diagnostic,
        }


def _parse_float(token: str) -> float:
    text = token.strip()
    if text == "":
        raise ValueError("empty numeric token")
    return float(text)


def parse_quantity(
    token: Any,
    dimension: str,
    unit: str | None,
    *,
    location: str | None = None,
) -> Quantity:
    """Parse a raw token for a linear dimension, preserving the representation."""

    raw_text = "" if token is None else str(token)
    loc = f" at {location}" if location else ""
    try:
        numeric = _parse_float(raw_text)
    except (TypeError, ValueError):
        return Quantity(
            dimension, raw_text, unit, None, None,
            f"non-numeric {dimension} token {raw_text!r}{loc}",
        )
    if unit is None:
        return Quantity(
            dimension, raw_text, None, None, None,
            f"missing unit for {dimension} token {raw_text!r}{loc}",
        )
    table = LINEAR_UNITS.get(dimension)
    if table is None or unit not in table:
        return Quantity(
            dimension, raw_text, unit, None, None,
            f"unknown {dimension} unit {unit!r}{loc}",
        )
    factor, canonical = table[unit]
    return Quantity(dimension, raw_text, unit, numeric * factor, canonical)


def parse_temperature(token: Any, unit: str | None, *, location: str | None = None) -> Quantity:
    """Parse an absolute temperature into kelvin."""

    raw_text = "" if token is None else str(token)
    loc = f" at {location}" if location else ""
    try:
        numeric = _parse_float(raw_text)
    except (TypeError, ValueError):
        return Quantity(
            "temperature", raw_text, unit, None, None,
            f"non-numeric temperature token {raw_text!r}{loc}",
        )
    if unit in ("K", "kelvin"):
        return Quantity("temperature", raw_text, unit, numeric, "K")
    if unit in ("C", "°C", "degC", "celsius"):
        return Quantity("temperature", raw_text, unit, numeric + 273.15, "K")
    if unit in ("F", "°F", "fahrenheit"):
        return Quantity("temperature", raw_text, unit, (numeric - 32.0) * 5.0 / 9.0 + 273.15, "K")
    return Quantity(
        "temperature", raw_text, unit, None, None,
        f"unknown temperature unit {unit!r}{loc}",
    )


def convert(value: float, dimension: str, unit: str) -> float:
    """Convert a value already in canonical units into a reporting unit."""

    if dimension == "temperature":
        if unit in ("K", "kelvin"):
            return value
        if unit in ("C", "°C", "degC"):
            return value - 273.15
        raise UnitError(f"unknown temperature unit {unit!r}")
    table = LINEAR_UNITS[dimension]
    if unit not in table:
        raise UnitError(f"unknown {dimension} unit {unit!r}")
    factor, canonical = table[unit]
    canonical_value = value
    # value is stored in the table canonical -> back out via factor
    return canonical_value / factor


def reporting(value: float, dimension: str) -> tuple[float, str]:
    """Return ``(value, unit)`` in the preferred reporting unit."""

    unit = CANONICAL_UNIT[dimension]
    return convert(value, dimension, unit), unit
