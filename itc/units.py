"""Unit systems, coordinate bases and raw-representation preserving conversion.

Domain rule: every domain quantity carries a unit (or coordinate/time basis).
The *raw representation* of an imported value is always preserved; conversion
to canonical units happens only in the derivation layer.  A value that cannot be
converted stays at its original location with a structured diagnostic instead of
being silently dropped or defaulted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CANONICAL = {
    "time": "s",
    "power": "uW",          # micro-watt (sample power curve)
    "energy": "uJ",          # micro-joule (integrated heat per injection)
    "volume": "uL",
    "concentration": "uM",  # micromolar
    "temperature": "K",
}

# multiplicative factor: value_in_canonical = factor * value_in_unit
_FACTORS = {
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "seconds"): 1.0,
    ("time", "min"): 60.0,
    ("time", "minute"): 60.0,
    ("time", "ms"): 1e-3,
    ("power", "uW"): 1.0,
    ("power", "uw"): 1.0,
    ("power", "µW"): 1.0,
    ("power", "uCal/s"): 4.184,
    ("power", "ucal/s"): 4.184,
    ("power", "cal/s"): 4.184e6,
    ("power", "mW"): 1000.0,
    ("energy", "uJ"): 1.0,
    ("energy", "µJ"): 1.0,
    ("energy", "mJ"): 1000.0,
    ("energy", "cal"): 4.184e6,
    ("energy", "mcal"): 4184.0,
    ("energy", "uCal"): 4.184,
    ("volume", "uL"): 1.0,
    ("volume", "µL"): 1.0,
    ("volume", "mL"): 1000.0,
    ("volume", "ml"): 1000.0,
    ("volume", "L"): 1e6,
    ("concentration", "uM"): 1.0,
    ("concentration", "µM"): 1.0,
    ("concentration", "mM"): 1000.0,
    ("concentration", "nM"): 1e-3,
    ("concentration", "M"): 1e6,
}


class UnitError(ValueError):
    """Raised when a unit is unknown for a quantity kind."""


@dataclass
class Diagnostic:
    """A value that could not be converted stays in place with this record."""

    code: str
    message: str
    location: str           # path-ish, e.g. materials/power_v2/rows/17/power
    raw: str | None = None
    kind: str | None = None
    unit: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "raw": self.raw,
            "kind": kind_unit(self.kind, self.unit) if self.kind else None,
        }


def kind_unit(kind: str | None, unit: str | None) -> str | None:
    if kind is None and unit is None:
        return None
    return f"{kind or '?'}[{unit or '?'}]"


def factor_for(kind: str, unit: str) -> float:
    if (kind, unit) not in _FACTORS:
        raise UnitError(f"unknown unit {unit!r} for kind {kind!r}")
    return _FACTORS[(kind, unit)]


def is_temperature(unit: str) -> bool:
    return unit in ("K", "degC", "C", "celsius", "degF", "F")


def convert(value: float, kind: str, unit: str) -> float:
    """Convert a numeric value to the canonical unit of its kind.

    Temperature needs an affine (offset) transform; everything else is scalar.
    """
    if kind == "temperature":
        if unit in ("K",):
            return float(value)
        if unit in ("degC", "C", "celsius"):
            return float(value) + 273.15
        if unit in ("degF", "F"):
            return (float(value) - 32.0) * 5.0 / 9.0 + 273.15
        raise UnitError(f"unknown temperature unit {unit!r}")
    return float(value) * factor_for(kind, unit)


@dataclass
class RawValue:
    """Imported scalar: original textual representation preserved verbatim."""

    raw: str
    kind: str
    unit: str
    location: str
    value: float | None = None
    diagnostic: Diagnostic | None = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "kind_unit": kind_unit(self.kind, self.unit),
            "canonical_unit": CANONICAL.get(self.kind),
            "location": self.location,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }


def parse_raw(raw, kind: str, unit: str, location: str) -> RawValue:
    """Parse while keeping the raw text; never raises for bad data.

    Unconvertible values return a RawValue with value=None and a diagnostic.
    """
    text = "" if raw is None else str(raw).strip()
    rv = RawValue(raw=text, kind=kind, unit=unit, location=location)
    if text == "":
        rv.diagnostic = Diagnostic("MISSING_VALUE", "empty value", location, text, kind, unit)
        return rv
    try:
        num = float(text)
    except ValueError:
        rv.diagnostic = Diagnostic("UNPARSEABLE_NUMBER", f"not a number: {text!r}", location, text, kind, unit)
        return rv
    try:
        rv.value = convert(num, kind, unit)
    except UnitError as exc:
        rv.diagnostic = Diagnostic("UNKNOWN_UNIT", str(exc), location, text, kind, unit)
    return rv


@dataclass
class Basis:
    """Coordinate / time basis for a sampled curve."""

    time_zero: str = "first_sample"   # first_sample | injection:ID | absolute
    time_unit: str = "s"
    power_unit: str = "uW"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "time_zero": self.time_zero,
            "time_unit": self.time_unit,
            "power_unit": self.power_unit,
            "notes": self.notes,
        }


@dataclass
class Quantity:
    """Normalized domain quantity with provenance back to its raw form."""

    value: float
    kind: str
    unit: str = field(default="")
    raw_location: str | None = None

    @property
    def canonical_unit(self) -> str:
        return CANONICAL[self.kind]

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.canonical_unit,
            "kind": self.kind,
            "source_unit": self.unit,
            "raw_location": self.raw_location,
        }
