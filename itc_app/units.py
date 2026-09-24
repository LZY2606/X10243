"""Quantity / unit handling.

Raw representations are preserved on ingestion. Conversion to the internal
canonical unit of a dimension happens only in the derivation layer
(`as_canonical`); values that cannot be converted stay in place with a
`QuantityError` diagnostic instead of being dropped or silently defaulted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Canonical units used inside derived calculations.
CANONICAL = {
    "power": "ucal/s",
    "time": "s",
    "volume": "L",
    "concentration": "mol/L",
    "temperature": "K",
    "energy": "cal",
    "amount": "mol",
}

# Multiplicative factors to canonical unit.
_FACTORS: dict[tuple[str, str], float] = {
    ("power", "cal/s"): 1e6,
    ("power", "ucal/s"): 1.0,
    ("power", "µcal/s"): 1.0,
    ("power", "uJ/s"): 1.0 / 4.184,
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "min"): 60.0,
    ("time", "ms"): 1e-3,
    ("volume", "L"): 1.0,
    ("volume", "mL"): 1e-3,
    ("volume", "uL"): 1e-6,
    ("volume", "µL"): 1e-6,
    ("concentration", "mol/L"): 1.0,
    ("concentration", "M"): 1.0,
    ("concentration", "mM"): 1e-3,
    ("concentration", "uM"): 1e-6,
    ("concentration", "µM"): 1e-6,
    ("concentration", "nM"): 1e-9,
    ("energy", "cal"): 1.0,
    ("energy", "ucal"): 1e-6,
    ("energy", "µcal"): 1e-6,
    ("energy", "kcal"): 1e3,
    ("energy", "J"): 1.0 / 4.184,
    ("amount", "mol"): 1.0,
    ("amount", "umol"): 1e-6,
    ("amount", "µmol"): 1e-6,
    ("amount", "nmol"): 1e-9,
}


class QuantityError(ValueError):
    """Raised when a raw value cannot be converted for its declared dimension."""

    def __init__(self, message: str, *, raw: Optional[str] = None, dimension: Optional[str] = None):
        super().__init__(message)
        self.raw = raw
        self.dimension = dimension
        self.message = message

    def diagnostic(self, path: str = "") -> dict:
        return {
            "path": path,
            "message": self.message,
            "raw": self.raw,
            "dimension": self.dimension,
        }


@dataclass(frozen=True)
class UnitSpec:
    dimension: str
    canonical: str


@dataclass
class Quantity:
    """A scalar with its original textual representation and unit.

    `raw_value` keeps exactly what the source file carried; `unit` is the
    declared unit. Nothing is converted until a derivation explicitly asks.
    """

    raw_value: str
    unit: str
    dimension: str
    value: Optional[float] = field(default=None, hash=False)
    canonical_value: Optional[float] = field(default=None, hash=False)
    error: Optional[str] = None

    def __post_init__(self):
        if self.value is None and self.error is None:
            try:
                self.value = float(self.raw_value)
            except (TypeError, ValueError):
                self.error = f"non-numeric scalar {self.raw_value!r} for {self.dimension}"

    def convert(self) -> float:
        """Return the value expressed in the dimension's canonical unit."""
        if self.canonical_value is not None:
            return self.canonical_value
        if self.error:
            raise QuantityError(self.error, raw=self.raw_value, dimension=self.dimension)
        if self.dimension == "temperature":
            converted = _convert_temperature(self.value, self.unit)
        else:
            key = (self.dimension, self.unit)
            if key not in _FACTORS:
                raise QuantityError(
                    f"unit {self.unit!r} is not defined for dimension {self.dimension!r}",
                    raw=self.raw_value,
                    dimension=self.dimension,
                )
            converted = self.value * _FACTORS[key]
        self.canonical_value = converted
        return converted

    def to_dict(self) -> dict:
        return {
            "raw": self.raw_value,
            "unit": self.unit,
            "dimension": self.dimension,
            "value": self.value,
            "canonical_unit": CANONICAL.get(self.dimension),
            "canonical_value": self.canonical_value,
            "error": self.error,
        }


def _convert_temperature(value: float, unit: str) -> float:
    if unit in ("K",):
        return value
    if unit in ("C", "°C", "degC"):
        return value + 273.15
    if unit in ("F", "°F"):
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    raise QuantityError(
        f"unit {unit!r} is not defined for dimension 'temperature'", raw=str(value), dimension="temperature"
    )


_TOKEN_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)\s*$")


def parse_quantity(text: object, dimension: str) -> Quantity:
    """Parse `"<number> <unit>"` (unit may be attached) preserving raw text."""
    if isinstance(text, (int, float)):
        raw = repr(text)
        unit = CANONICAL.get(dimension, "")
        return Quantity(raw_value=raw, unit=unit, dimension=dimension, value=float(text))
    raw = str(text)
    match = _TOKEN_RE.match(raw)
    if not match:
        return Quantity(raw_value=raw, unit="", dimension=dimension,
                        error=f"cannot parse {raw!r} as {dimension}")
    number_text, unit = match.group(1), match.group(2)
    if not unit:
        unit = CANONICAL.get(dimension, "")
    try:
        number = float(number_text)
    except ValueError:
        return Quantity(raw_value=raw, unit=unit, dimension=dimension,
                        error=f"non-numeric scalar {number_text!r}")
    return Quantity(raw_value=raw, unit=unit, dimension=dimension, value=number)


def try_convert(text: object, dimension: str, path: str = "") -> tuple[Optional[float], Optional[dict]]:
    q = parse_quantity(text, dimension)
    try:
        return q.convert(), None
    except QuantityError as exc:
        return None, exc.diagnostic(path)


def energy_text(cal: float) -> str:
    return f"{cal:.6g} cal"
