"""Quantities with units and coordinate/time bases.

Raw material values keep their original string representation (``raw``).  Only
the derived layer normalises them.  A value that cannot be converted is left at
its original position together with a diagnostic instead of being dropped or
replaced by a default.

Canonical internal bases used by the engine:

    time          seconds (s)
    power         microcalorie / second (ucal/s) -- MicroCal ITC convention
    energy        microcalorie (ucal)
    volume        microlitre (uL)
    concentration millimolar (mM)
    temperature   kelvin (K)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic

# factor maps unit -> (canonical unit, multiplicative factor, additive offset)
_UNITS: dict[str, tuple[str, float, float]] = {}
_DIMENSIONS: dict[str, str] = {}


def _register(dimension: str, canonical: str, table: dict[str, float],
              offsets: dict[str, float] | None = None) -> None:
    offsets = offsets or {}
    for unit, factor in table.items():
        _UNITS[unit] = (canonical, factor, offsets.get(unit, 0.0))
        _DIMENSIONS[unit] = dimension


_register("time", "s", {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
})
_register("power", "ucal/s", {
    "ucal/s": 1.0, "µcal/s": 1.0, "ucal/sec": 1.0,
    "mcal/s": 1e3, "cal/s": 1e6, "kcal/s": 1e9,
    "uw": 0.2390057361376673, "µw": 0.2390057361376673,
    "mw": 239.0057361376673,
})
_register("energy", "ucal", {
    "ucal": 1.0, "µcal": 1.0, "mcal": 1e3, "cal": 1e6,
    "kcal": 1e9, "j": 0.2390057361376673e6, "joule": 0.2390057361376673e6,
    "mj": 239.0057361376673, "kj": 239005.7361376673,
})
_register("volume", "uL", {
    "ul": 1.0, "µl": 1.0, "ml": 1e3, "l": 1e6, "liter": 1e6,
})
_register("concentration", "mM", {
    "mm": 1.0, "mmol/l": 1.0, "mmolar": 1.0,
    "um": 1e-3, "µm": 1e-3, "microm": 1e-3,
    "nm": 1e-6, "pm": 1e-9,
    "m": 1e3, "molar": 1e3,
})
_register("temperature", "K", {
    "k": 1.0,
    "c": 1.0, "°c": 1.0, "celsius": 1.0,
}, offsets={"c": 273.15, "°c": 273.15, "celsius": 273.15})

_NUMBER_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)\s*$")


def normalise_unit(unit: str) -> str:
    return unit.strip().replace("μ", "u").lower()


@dataclass
class Quantity:
    """A measurement that remembers how it was written."""
    value: float | None                 # canonical value; None if unconvertible
    raw_text: str
    unit_raw: str
    unit_canonical: str | None
    dimension: str | None
    diagnostics: list[Diagnostic]

    @property
    def convertible(self) -> bool:
        return self.value is not None and not any(
            d.severity == "error" for d in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw": self.raw_text,
            "unit_raw": self.unit_raw,
            "unit_canonical": self.unit_canonical,
            "dimension": self.dimension,
            "convertible": self.convertible,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def parse_quantity(raw_text: Any, *, expected_dimension: str | None = None,
                   location: dict[str, Any] | None = None) -> Quantity:
    location = dict(location or {})
    diagnostics: list[Diagnostic] = []
    text = str(raw_text).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        diagnostics.append(Diagnostic(
            "error", "not_a_number",
            f"Value {text!r} is not a number with optional unit.",
            location, raw_text))
        return Quantity(None, text, "", None, None, diagnostics)

    number_text, unit_text = match.groups()
    number = float(number_text)
    unit_key = normalise_unit(unit_text) if unit_text else ""

    if not unit_key:
        # A bare number is kept raw but cannot be normalised without a unit.
        diagnostics.append(Diagnostic(
            "error", "missing_unit",
            f"Value {text!r} has no unit; it cannot be normalised.",
            location, raw_text))
        return Quantity(None, text, "", None, None, diagnostics)

    if unit_key not in _UNITS:
        diagnostics.append(Diagnostic(
            "error", "unknown_unit",
            f"Unit {unit_text!r} is not recognised.",
            location, raw_text))
        return Quantity(None, text, unit_text, None, None, diagnostics)

    canonical_unit, factor, offset = _UNITS[unit_key]
    dimension = _DIMENSIONS[unit_key]
    value = number * factor + offset
    if math.isnan(value) or math.isinf(value):
        diagnostics.append(Diagnostic(
            "error", "non_finite_value",
            f"Value {text!r} normalises to a non-finite number.",
            location, raw_text))
        return Quantity(None, text, unit_text, canonical_unit, dimension, diagnostics)

    if expected_dimension is not None and dimension != expected_dimension:
        diagnostics.append(Diagnostic(
            "error", "dimension_mismatch",
            f"Unit {unit_text!r} is a {dimension} unit; expected {expected_dimension}.",
            location, raw_text))
        return Quantity(None, text, unit_text, canonical_unit, dimension, diagnostics)

    return Quantity(value, text, unit_text, canonical_unit, dimension, diagnostics)


def convert(value: float, unit_raw: str, *, expected_dimension: str | None = None,
            location: dict[str, Any] | None = None) -> float:
    """Strict conversion used by the engine; raises rather than guessing."""
    combined = f"{value} {unit_raw}" if unit_raw else str(value)
    quantity = parse_quantity(combined, expected_dimension=expected_dimension,
                              location=location)
    if not quantity.convertible:
        from .errors import SemanticUnsolvable
        raise SemanticUnsolvable(
            f"Cannot convert {combined!r}",
            locations=[location] if location else [],
            diagnostics=[d.to_dict() for d in quantity.diagnostics])
    assert quantity.value is not None
    return quantity.value
