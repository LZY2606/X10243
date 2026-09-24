"""Unit algebra for ITC quantities.

Raw layers keep the exact representation that arrived with the material
(``raw`` string + ``unit``); only the derived layer asks for ``base`` values.
Values that cannot be converted stay in place and attach a diagnostic; they
are never dropped to satisfy a plotter.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import Diagnostic

# dimension -> {unit label: (factor to base, additive offset)}
_REGISTRY: dict[str, dict[str, tuple[float, float]]] = {
    "time": {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "min": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hour": 3600.0,
    },
    "power": {  # base: microwatt (uW) — ITC convention
        "uW": 1.0,
        "µW": 1.0,
        "mW": 1000.0,
        "W": 1_000_000.0,
        "cal/s": 4.184 * 1_000_000.0,
    },
    "volume": {  # base: microliter (uL)
        "uL": 1.0,
        "µL": 1.0,
        "mL": 1000.0,
        "L": 1_000_000.0,
    },
    "concentration": {  # base: micromolar (uM)
        "uM": 1.0,
        "µM": 1.0,
        "mM": 1000.0,
        "M": 1_000_000.0,
        "nM": 0.001,
    },
    "temperature": {  # base: kelvin, label in celsius
        "K": (1.0, 0.0),
        "degC": (1.0, 273.15),
        "C": (1.0, 273.15),
        "°C": (1.0, 273.15),
        "degF": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    },
    "energy": {  # base: microjoule (uJ)
        "uJ": 1.0,
        "µJ": 1.0,
        "mJ": 1000.0,
        "J": 1_000_000.0,
        "cal": 4.184 * 1_000_000.0,
        "kcal": 4.184 * 1_000_000_000.0,
    },
    "molar_heat": {  # base: kcal/mol (reporting convention)
        "kcal/mol": 1.0,
        "cal/mol": 0.001,
        "kJ/mol": 1.0 / 4.184,
        "J/mol": 0.001 / 4.184,
    },
}

BASE_UNIT = {
    "time": "s",
    "power": "uW",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
    "energy": "uJ",
    "molar_heat": "kcal/mol",
}


@dataclass
class Quantity:
    """A value carrying both its original textual representation and base value."""

    raw: str
    unit: str
    dimension: str
    base: float | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "dimension": self.dimension,
            "base": self.base,
            "base_unit": BASE_UNIT[self.dimension],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def convert(value: float, unit: str, dimension: str) -> float:
    """Convert a numerical value of ``unit`` into the dimension's base unit."""
    units = _REGISTRY[dimension]
    if unit not in units:
        raise KeyError(f"unknown unit {unit!r} for dimension {dimension!r}")
    factor, offset = units[unit]
    return value * factor + offset


def parse_quantity(raw: Any, unit: str, dimension: str, location: dict[str, Any] | None = None) -> Quantity:
    """Parse one scalar, preserving the raw token even when conversion fails."""
    location = dict(location or {})
    q = Quantity(raw=str(raw), unit=unit, dimension=dimension)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        q.diagnostics.append(
            Diagnostic(
                "UNPARSEABLE_VALUE",
                f"value {raw!r} is not numeric for {dimension}",
                {**location, "raw": str(raw)},
            )
        )
        return q
    if not math.isfinite(value):
        q.diagnostics.append(
            Diagnostic(
                "NON_FINITE_VALUE",
                f"value {raw!r} is not finite",
                {**location, "raw": str(raw)},
            )
        )
        return q
    if dimension not in _REGISTRY:
        q.diagnostics.append(
            Diagnostic("UNKNOWN_DIMENSION", f"no registry for {dimension!r}", location)
        )
        return q
    if unit not in _REGISTRY[dimension]:
        q.diagnostics.append(
            Diagnostic(
                "UNKNOWN_UNIT",
                f"unit {unit!r} is not registered for {dimension}; raw value retained",
                {**location, "unit": unit},
            )
        )
        return q
    q.base = convert(value, unit, dimension)
    return q


def convert_series(values: list[float], unit: str, dimension: str) -> list[float]:
    factor, offset = _REGISTRY[dimension][unit]
    return [v * factor + offset for v in values]


def is_known_unit(unit: str, dimension: str) -> bool:
    return dimension in _REGISTRY and unit in _REGISTRY[dimension]
