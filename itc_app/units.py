"""Unit conversion for ITC domain values.

Original representations are preserved by callers. This module only normalizes
copies in the derived layer. Unconvertible inputs raise UnitDiagnostic with a
stable location instead of being silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitDiagnostic(Exception):
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.code} at {self.location}: {self.message}"


_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0}
_POWER = {"µW": 1.0, "uW": 1.0, "microwatt": 1.0, "mW": 1000.0, "nW": 0.001}
_VOLUME = {"µL": 1.0, "uL": 1.0, "microliter": 1.0, "mL": 1000.0, "L": 1_000_000.0}
_CONCENTRATION = {
    "µM": 1.0,
    "uM": 1.0,
    "micromolar": 1.0,
    "mM": 1000.0,
    "nM": 0.001,
    "M": 1_000_000.0,
}
_TEMPERATURE = {"°C": "C", "C": "C", "degC": "C", "K": "K", "°F": "F", "F": "F"}
_ENERGY = {"µJ": 1.0, "uJ": 1.0, "microjoule": 1.0, "mJ": 1000.0, "cal": 4.184, "kcal": 4184.0}


def _number(value, location: str) -> float:
    if isinstance(value, bool):
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, f"Boolean is not a numeric value: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, f"Cannot interpret numeric value {value!r}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, f"Numeric value must be finite: {value!r}")
    return number


def _convert(value, unit: str, table, normalized_unit: str, location: str) -> float:
    number = _number(value, location)
    if unit not in table:
        raise UnitDiagnostic(
            "SEMANTIC_NO_SOLUTION",
            location,
            f"Unsupported {normalized_unit} unit {unit!r}; original value remains {value!r} {unit}",
        )
    return number * table[unit]


def time_s(value, unit: str, location: str) -> float:
    return _convert(value, unit, _TIME, "time", location)


def power_uW(value, unit: str, location: str) -> float:
    return _convert(value, unit, _POWER, "power", location)


def volume_uL(value, unit: str, location: str) -> float:
    converted = _convert(value, unit, _VOLUME, "volume", location)
    if converted <= 0:
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, "Volume must be greater than zero")
    return converted


def concentration_uM(value, unit: str, location: str) -> float:
    converted = _convert(value, unit, _CONCENTRATION, "concentration", location)
    if converted < 0:
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, "Concentration cannot be negative")
    return converted


def energy_uJ(value, unit: str, location: str) -> float:
    return _convert(value, unit, _ENERGY, "energy", location)


def temperature_C(value, unit: str, location: str) -> float:
    number = _number(value, location)
    kind = _TEMPERATURE.get(unit)
    if kind is None:
        raise UnitDiagnostic("SEMANTIC_NO_SOLUTION", location, f"Unsupported temperature unit {unit!r}")
    if kind == "C":
        return number
    if kind == "K":
        return number - 273.15
    return (number - 32.0) * 5.0 / 9.0


def temperature_display(value_c: float) -> dict:
    return {"value": value_c, "unit": "°C", "basis": "normalized derived layer"}
