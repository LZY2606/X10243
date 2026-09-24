"""Quantities carry a raw representation plus a coordinate/unit basis.

Import keeps the original text at its origin; normalisation happens only in the
derived layer. Values that cannot be converted stay in place and receive a
diagnostic rather than being dropped or replaced with a default.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SemanticImpossible

# canonical SI unit -> multipliers (value * m = canonical value)
_UNITS: dict[str, dict[str, float]] = {
    "power": {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "µW": 1e-6},
    "energy": {"J": 1.0, "mJ": 1e-3, "uJ": 1e-6, "cal": 4.184, "kcal": 4184.0},
    "molar_heat": {"J/mol": 1.0, "kJ/mol": 1e3, "kcal/mol": 4184.0, "cal/mol": 4.184},
    "time": {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0},
    "volume": {"L": 1.0, "mL": 1e-3, "uL": 1e-6, "µL": 1e-6},
    "concentration": {
        "M": 1.0, "mol/L": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9,
    },
    "temperature": {"K": 1.0, "C": 1.0, "degC": 1.0, "°C": 1.0},
}

CANONICAL = {
    "power": "W",
    "energy": "J",
    "molar_heat": "J/mol",
    "time": "s",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
}


def known_units(dim: str) -> set[str]:
    return set(_UNITS.get(dim, {}))


def multiplier(dim: str, unit: str) -> float:
    if dim not in _UNITS:
        raise SemanticImpossible(f"unknown dimension '{dim}'")
    if unit not in _UNITS[dim]:
        raise SemanticImpossible(
            f"unit '{unit}' is not a registered {dim} unit",
            diagnostics=[{"code": "unknown_unit", "dimension": dim, "unit": unit}],
        )
    return _UNITS[dim][unit]


def to_canonical(value: float, dim: str, unit: str) -> float:
    if dim == "temperature":
        if unit == "K":
            return value
        return value + 273.15
    return value * multiplier(dim, unit)


def from_canonical(value: float, dim: str, unit: str) -> float:
    if dim == "temperature":
        return value if unit == "K" else value - 273.15
    return value / multiplier(dim, unit)


@dataclass(frozen=True)
class Quantity:
    """A scalar with the representation it arrived in and its canonical value."""

    value: float
    unit: str
    dimension: str
    raw: str | None = None

    @staticmethod
    def parse(raw: str, dim: str, *, default_unit: str | None = None) -> "Quantity":
        text = (raw or "").strip()
        unit = default_unit or CANONICAL[dim]
        num_part = text
        if text:
            for u in sorted(known_units(dim), key=len, reverse=True):
                if text.endswith(u):
                    unit = u
                    num_part = text[: -len(u)].strip()
                    break
        try:
            value = float(num_part)
        except (ValueError, TypeError):
            q = Quantity(float("nan"), unit, dim, raw=raw)
            raise SemanticImpossible(
                f"cannot convert '{raw}' to {dim}",
                diagnostics=[{"code": "unconvertible_value", "dimension": dim, "raw": raw, "at": q.origin}],
            )
        canon = to_canonical(value, dim, unit)
        return Quantity(canon, CANONICAL[dim], dim, raw=raw)

    @property
    def origin(self) -> str:
        return self.raw if self.raw is not None else f"{self.value}"


def format_quantity(canon: float, dim: str, unit: str | None = None, ndigits: int = 6) -> str:
    unit = unit or CANONICAL[dim]
    shown = from_canonical(canon, dim, unit)
    return f"{shown:.{ndigits}g} {unit}"
