from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    location: dict[str, Any]
    raw_value: Any = None


@dataclass(frozen=True)
class Measurement:
    raw_text: str
    value: float
    unit: str
    normalized_value: float
    normalized_unit: str
    coordinate: str | None = None


_CONVERSIONS = {
    "time": {"s": (1.0, "s"), "sec": (1.0, "s"), "min": (60.0, "s")},
    "power": {
        "µcal/s": (1.0, "µcal/s"),
        "ucal/s": (1.0, "µcal/s"),
        "cal/s": (1_000_000.0, "µcal/s"),
    },
    "volume": {
        "µL": (1.0, "µL"),
        "uL": (1.0, "µL"),
        "mL": (1000.0, "µL"),
        "L": (1_000_000.0, "µL"),
    },
    "concentration": {
        "µM": (1.0, "µM"),
        "uM": (1.0, "µM"),
        "mM": (1000.0, "µM"),
        "M": (1_000_000.0, "µM"),
    },
    "heat": {
        "µcal": (1.0, "µcal"),
        "ucal": (1.0, "µcal"),
        "cal": (1_000_000.0, "µcal"),
    },
}


def parse_number(raw: object) -> float:
    if raw is None:
        raise ValueError("missing value")
    text = str(raw).strip()
    if not text:
        raise ValueError("empty value")
    return float(text)


def measurement(
    kind: str,
    raw: object,
    unit: str,
    *,
    location: dict[str, Any] | None = None,
    coordinate: str | None = None,
) -> tuple[Measurement | None, Diagnostic | None]:
    try:
        number = parse_number(raw)
    except (TypeError, ValueError):
        return None, Diagnostic(
            "unconvertible_measurement",
            f"Value {raw!r} cannot be converted for {kind}.",
            location or {"field": kind},
            raw,
        )
    units = _CONVERSIONS[kind]
    if unit not in units:
        return None, Diagnostic(
            "unsupported_unit",
            f"Unit {unit!r} is not supported for {kind}.",
            location or {"field": kind},
            raw,
        )
    factor, normalized_unit = units[unit]
    return (
        Measurement(str(raw), number, unit, number * factor, normalized_unit, coordinate),
        None,
    )


def parse_temperature(raw: object, unit: str, *, location: dict[str, Any] | None = None):
    try:
        number = parse_number(raw)
    except (TypeError, ValueError):
        return None, Diagnostic("unconvertible_measurement", "Temperature cannot be converted.", location or {}, raw)
    if unit in {"K", "kelvin"}:
        kelvin = number
        canonical = "K"
    elif unit in {"°C", "C", "degC"}:
        kelvin = number + 273.15
        canonical = "°C"
    else:
        return None, Diagnostic("unsupported_unit", f"Unsupported temperature unit {unit!r}.", location or {}, raw)
    return Measurement(str(raw), number, unit, kelvin, "K", canonical), None


def heat_microcal_to_cal(value: float) -> float:
    return value / 1_000_000.0


def micro_liter_to_liter(value: float) -> float:
    return value / 1_000_000.0


def micro_molar_to_molar(value: float) -> float:
    return value / 1_000_000.0
