"""Dimensioned quantities with preserved original representations.

Every imported domain value keeps three things:

* ``raw``  - exactly the text/token that appeared in the source file;
* ``unit`` - the unit the source declared (or the dataset coordinate basis);
* ``value`` - the value in SI/internal units, or ``None`` when conversion is
  impossible.  Unconvertible values are never dropped: they stay in place and
  carry a diagnostic.

Normalisation only happens in the derived layer (``engine``). Materials only
store the original representation plus an optional parsed value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Canonical internal bases ------------------------------------------------
# time: seconds (s); volume: litre (L); concentration: mol/L (M);
# temperature: kelvin (K) (conversions keep the Celsius origin);
# power: watt (W); energy: joule (J).

_UNIT_FACTORS: dict[tuple[str, str], float] = {
    # (dimension, unit token) -> multiplier to internal unit
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "second"): 1.0,
    ("time", "seconds"): 1.0,
    ("time", "min"): 60.0,
    ("time", "minute"): 60.0,
    ("time", "minutes"): 60.0,
    ("volume", "L"): 1.0,
    ("volume", "l"): 1.0,
    ("volume", "mL"): 1e-3,
    ("volume", "ml"): 1e-3,
    ("volume", "uL"): 1e-6,
    ("volume", "µL"): 1e-6,
    ("volume", "ul"): 1e-6,
    ("concentration", "M"): 1.0,
    ("concentration", "mol/L"): 1.0,
    ("concentration", "mM"): 1e-3,
    ("concentration", "mm"): 1e-3,
    ("concentration", "uM"): 1e-6,
    ("concentration", "µM"): 1e-6,
    ("concentration", "um"): 1e-6,
    ("concentration", "nM"): 1e-9,
    ("power", "W"): 1.0,
    ("power", "w"): 1.0,
    ("power", "mW"): 1e-3,
    ("power", "mw"): 1e-3,
    ("power", "uW"): 1e-6,
    ("power", "µW"): 1e-6,
    ("power", "uw"): 1e-6,
    ("energy", "J"): 1.0,
    ("energy", "j"): 1.0,
    ("energy", "mJ"): 1e-3,
    ("energy", "cal"): 4.184,
    ("energy", "kcal"): 4184.0,
}

# Temperature needs origin handling, hence a separate map.
_TEMPERATURE_UNITS = {"K": "K", "kelvin": "K", "C": "C", "celsius": "C",
                      "degC": "C", "°C": "C"}

INTERNAL_UNIT = {
    "time": "s",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
    "power": "W",
    "energy": "J",
}


class QuantityError(ValueError):
    """Raised for semantically invalid unit declarations."""


@dataclass(frozen=True)
class Quantity:
    """A single dimensioned scalar preserving its source spelling."""

    raw: str
    unit: str
    value: Optional[float]
    dimension: Optional[str] = None
    diagnostic: Optional[str] = None
    basis: Optional[str] = None  # coordinate/time basis, e.g. "seconds since injection_1"

    @property
    def ok(self) -> bool:
        return self.value is not None and self.diagnostic is None

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "value": self.value,
            "dimension": self.dimension,
            "diagnostic": self.diagnostic,
            "basis": self.basis,
        }


def parse_quantity(raw: object, dimension: str, default_unit: Optional[str] = None,
                   basis: Optional[str] = None) -> Quantity:
    """Parse ``raw`` into the internal unit of ``dimension``.

    A numeric token may omit a unit only when ``default_unit`` is supplied.
    Anything that cannot be parsed/convert is returned as a failed Quantity
    with a diagnostic rather than raising: importers must not lose the cell.
    """
    text = "" if raw is None else str(raw).strip()
    if text == "":
        return Quantity(raw=text, unit=default_unit or "", value=None,
                        dimension=dimension, diagnostic="empty value", basis=basis)

    unit_token, number_text = _split_value_unit(text, default_unit)
    if number_text is None:
        return Quantity(raw=text, unit=unit_token or "", value=None,
                        dimension=dimension,
                        diagnostic=f"unparseable {dimension} value: {text!r}", basis=basis)
    try:
        number = float(number_text)
    except ValueError:
        return Quantity(raw=text, unit=unit_token or "", value=None,
                        dimension=dimension,
                        diagnostic=f"non-numeric {dimension}: {number_text!r}", basis=basis)

    if dimension == "temperature":
        return _parse_temperature(text, unit_token, number, basis)

    if unit_token is None:
        return Quantity(raw=text, unit="", value=None, dimension=dimension,
                        diagnostic=f"missing unit for {dimension}", basis=basis)
    factor = _UNIT_FACTORS.get((dimension, unit_token))
    if factor is None:
        return Quantity(raw=text, unit=unit_token, value=None, dimension=dimension,
                        diagnostic=f"unknown {dimension} unit {unit_token!r}", basis=basis)
    return Quantity(raw=text, unit=unit_token, value=number * factor,
                    dimension=dimension, basis=basis)


def _split_value_unit(text: str, default_unit: Optional[str]):
    # Forms: "12.3", "12.3uL", "12.3 uL", "1.5e-3 M"
    cut = len(text)
    seen_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            seen_space = True
            cut = i
            break
        if i > 0 and (ch.isalpha() or ch in "µ°μ"):
            cut = i
            break
    num_text = text[:cut].strip()
    unit_text = text[cut:].strip()
    if unit_text == "" and default_unit:
        return default_unit, num_text
    if unit_text == "":
        return None, num_text
    return unit_text, num_text


def _parse_temperature(text: str, unit_token: Optional[str], number: float,
                       basis: Optional[str]) -> Quantity:
    if unit_token is None:
        return Quantity(raw=text, unit="", value=None, dimension="temperature",
                        diagnostic="missing unit for temperature", basis=basis)
    kind = _TEMPERATURE_UNITS.get(unit_token)
    if kind is None:
        return Quantity(raw=text, unit=unit_token, value=None, dimension="temperature",
                        diagnostic=f"unknown temperature unit {unit_token!r}", basis=basis)
    value = number if kind == "K" else number + 273.15
    return Quantity(raw=text, unit=unit_token, value=value,
                    dimension="temperature", basis=basis)


def convert(value: float, dimension: str, unit: str) -> float:
    """Convert an internal-unit numeric value back into ``unit`` (for display)."""
    if dimension == "temperature":
        kind = _TEMPERATURE_UNITS.get(unit)
        if kind == "K":
            return value
        if kind == "C":
            return value - 273.15
        raise QuantityError(f"unknown temperature unit {unit!r}")
    factor = _UNIT_FACTORS.get((dimension, unit))
    if factor is None:
        raise QuantityError(f"unknown {dimension} unit {unit!r}")
    return value / factor


def format_internal(value: Optional[float], dimension: str, precision: int = 5) -> str:
    """Render an internal-unit value with its canonical unit."""
    if value is None:
        return "—"
    unit = INTERNAL_UNIT[dimension]
    if dimension == "temperature":
        return f"{value - 273.15:.{precision}g} °C ({value:.{precision}g} K)"
    return f"{value:.{precision}g} {unit}"
