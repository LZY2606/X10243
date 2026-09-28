from __future__ import annotations

from dataclasses import dataclass

from .errors import Diagnostic, SemanticImpossibleError


TIME_TO_SECOND = {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0}
POWER_TO_MICROWATT = {"µW": 1.0, "uW": 1.0, "μW": 1.0, "mW": 1000.0, "W": 1_000_000.0}
VOLUME_TO_MICROLITRE = {"µL": 1.0, "uL": 1.0, "μL": 1.0, "mL": 1000.0, "L": 1_000_000.0}
CONCENTRATION_TO_MM = {"mM": 1.0, "mmol/L": 1.0, "µM": 0.001, "uM": 0.001, "μM": 0.001, "M": 1000.0}
TEMPERATURE_TO_CELSIUS = {"°C": 1.0, "C": 1.0, "K": None, "K_offset": 273.15}
ENERGY_TO_MICROJOULE = {"µJ": 1.0, "uJ": 1.0, "μJ": 1.0, "mJ": 1000.0, "J": 1_000_000.0, "cal": 4.184, "kcal": 4184.0}


@dataclass(frozen=True)
class NumericWithUnit:
    value: float
    unit: str
    raw: object
    canonical_value: float
    canonical_unit: str

    def display(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
        }


def number(value: object, location: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticImpossibleError(
            "value cannot be converted",
            [Diagnostic("UNCONVERTIBLE_VALUE", f"cannot convert {value!r} to number", location)],
        ) from exc
    if result != result:
        raise SemanticImpossibleError(
            "NaN is not accepted",
            [Diagnostic("UNCONVERTIBLE_VALUE", "NaN is not a valid measurement", location)],
        )
    return result


def convert(
    value: object,
    unit: object,
    factors: dict[str, float | None],
    canonical_unit: str,
    location: str,
) -> NumericWithUnit:
    numeric = number(value, location)
    if not isinstance(unit, str) or unit not in factors:
        raise SemanticImpossibleError(
            "unknown unit",
            [Diagnostic("UNKNOWN_UNIT", f"unsupported unit {unit!r}", location)],
        )
    factor = factors[unit]
    canonical = numeric + TEMPERATURE_TO_CELSIUS["K_offset"] if factor is None and unit == "K" else numeric * factor
    return NumericWithUnit(numeric, unit, value, canonical, canonical_unit)


def time_seconds(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, TIME_TO_SECOND, "s", location)


def power_microwatt(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, POWER_TO_MICROWATT, "µW", location)


def volume_microlitre(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, VOLUME_TO_MICROLITRE, "µL", location)


def concentration_mm(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, CONCENTRATION_TO_MM, "mM", location)


def temperature_celsius(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, TEMPERATURE_TO_CELSIUS, "°C", location)


def energy_microjoule(value: object, unit: object, location: str) -> NumericWithUnit:
    return convert(value, unit, ENERGY_TO_MICROJOULE, "µJ", location)
