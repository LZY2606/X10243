"""Raw values with units and a deliberately small conversion engine.

The import layer keeps the *original representation* (``{"value": "0.32",
"unit": "mM"}``).  Only the derived layer normalises via :func:`normalise`,
which returns ``(value_in_si, canonical_unit, None)`` or
``(None, original_unit, Diagnostic)``.  A value that cannot be converted stays
in its original location with a diagnostic - it is never dropped, never
silently replaced by a default.

The supported domains are exactly the ones an ITC review needs:

* time        -> seconds (s), with explicit reference epoch string
* power       -> cal/s, µcal/s, µW
* energy      -> cal, µcal, J
* volume      -> L, mL, µL
* concentration -> M, mM, µM
* temperature -> K, °C / degC (kept in K downstream, original retained)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic

# multiplicative factors to the canonical SI unit of each dimension
_TO_SI: dict[tuple[str, str], float] = {
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "second"): 1.0,
    ("time", "ms"): 1e-3,
    ("time", "min"): 60.0,
    ("power", "cal/s"): 1.0,
    ("power", "ucal/s"): 1e-6,
    ("power", "µcal/s"): 1e-6,
    ("power", "w"): 1 / 4.184,
    ("power", "W"): 1 / 4.184,
    ("power", "uw"): 1e-6 / 4.184,
    ("power", "µw"): 1e-6 / 4.184,
    ("energy", "cal"): 1.0,
    ("energy", "ucal"): 1e-6,
    ("energy", "µcal"): 1e-6,
    ("energy", "kcal"): 1e3,
    ("energy", "j"): 1 / 4.184,
    ("energy", "J"): 1 / 4.184,
    ("volume", "l"): 1.0,
    ("volume", "L"): 1.0,
    ("volume", "ml"): 1e-3,
    ("volume", "mL"): 1e-3,
    ("volume", "ul"): 1e-6,
    ("volume", "µL"): 1e-6,
    ("concentration", "m"): 1.0,
    ("concentration", "M"): 1.0,
    ("concentration", "mm"): 1e-3,
    ("concentration", "mM"): 1e-3,
    ("concentration", "um"): 1e-6,
    ("concentration", "µM"): 1e-6,
    ("concentration", "nm"): 1e-9,
    ("concentration", "nM"): 1e-9,
}

CANONICAL_UNIT = {
    "time": "s",
    "power": "cal/s",
    "energy": "cal",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
}

_DIMENSIONS = set(CANONICAL_UNIT)


def raw_value(obj: Any, *, location: str = "", dimension: str | None = None) -> "RawValue":
    """Parse an imported scalar that may carry a unit.

    Accepts either ``{"value": ..., "unit": ...}`` or a bare number/string.
    The original lexical form is stored verbatim in ``raw``.
    """
    if isinstance(obj, RawValue):
        return obj
    if isinstance(obj, dict):
        if "value" not in obj:
            return RawValue(None, "", obj, location or "$",
                            Diagnostic("unit:missing_value", "raw value object lacks 'value'",
                                       location or "$", obj))
        value = obj["value"]
        unit = obj.get("unit", "")
        ref = obj.get("reference")
        return RawValue(value, unit, obj, location or "$", None, reference=ref)
    return RawValue(obj, "", obj, location or "$", None)


@dataclass
class RawValue:
    """A scalar exactly as imported plus optional normalised form."""

    value: Any
    unit: str
    raw: Any
    location: str = "$"
    diagnostic: Diagnostic | None = None
    reference: str | None = None

    @property
    def is_convertible(self) -> bool:
        return self.diagnostic is None

    def as_display(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "reference": self.reference,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }


def normalise(rv: RawValue, dimension: str) -> tuple[float | None, str, Diagnostic | None]:
    """Return ``(si_value, canonical_unit, diagnostic)``."""
    if dimension not in _DIMENSIONS:
        return None, CANONICAL_UNIT.get(dimension, rv.unit), Diagnostic(
            "unit:unknown_dimension", f"unknown dimension {dimension!r}", rv.location, rv.raw)
    if rv.diagnostic is not None:
        return None, rv.unit or CANONICAL_UNIT[dimension], rv.diagnostic
    try:
        number = float(rv.value)
    except (TypeError, ValueError):
        diag = Diagnostic("unit:not_a_number",
                          f"value {rv.value!r} cannot be interpreted as a number",
                          rv.location, rv.raw)
        return None, rv.unit or CANONICAL_UNIT[dimension], diag
    if number != number or number in (float("inf"), float("-inf")):
        diag = Diagnostic("unit:not_finite", "value must be finite", rv.location, rv.raw)
        return None, rv.unit or CANONICAL_UNIT[dimension], diag
    if dimension == "temperature":
        unit = (rv.unit or "K").strip()
        low = unit.lower()
        if low in ("k", "kelvin"):
            return number, "K", None
        if low in ("c", "°c", "degc", "celsius"):
            return number + 273.15, "K", None
        diag = Diagnostic("unit:unknown_temperature",
                          f"temperature unit {rv.unit!r} is not K or °C", rv.location, rv.raw)
        return None, rv.unit, diag
    key = (dimension, rv.unit)
    if key not in _TO_SI:
        diag = Diagnostic("unit:unsupported",
                          f"unit {rv.unit!r} is not supported for {dimension}",
                          rv.location, rv.raw)
        return None, rv.unit or CANONICAL_UNIT[dimension], diag
    return number * _TO_SI[key], CANONICAL_UNIT[dimension], None


def require_number(rv: RawValue, dimension: str) -> float:
    """Normalise or raise :class:`SemanticInfeasible` carrying the location."""
    from .errors import SemanticInfeasible
    value, _unit, diag = normalise(rv, dimension)
    if diag is not None:
        raise SemanticInfeasible(diag.message, [diag])
    assert value is not None
    return value


def format_energy(cal: float) -> str:
    """Human friendly µcal string (canonical storage remains cal)."""
    return f"{cal * 1e6:.4g} µcal"
