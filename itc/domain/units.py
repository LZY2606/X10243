"""Unit / coordinate metadata.

Domain quantities carry their unit or coordinate basis explicitly. Raw
representations are preserved by the parser; only the derived layer performs
normalisation. A value that cannot be converted is returned in place with a
diagnostic rather than being dropped or defaulted.
"""
from __future__ import annotations

from dataclasses import dataclass

CANONICAL = {
    "power": "ucal/s",
    "heat": "ucal",
    "time": "s",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
}

# multiplicative factors to canonical unit; temperature is handled specially
_UNIT_FACTORS = {
    "power": {"ucal/s": 1.0, "cal/s": 1e6, "ucal/sec": 1.0, "mw": 239.005736},
    "heat": {"ucal": 1.0, "cal": 1e6, "j": 2.39005736e5, "mj": 239.005736},
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0, "minute": 60.0},
    "volume": {"ul": 1.0, "µl": 1.0, "ml": 1000.0, "l": 1e6},
    "concentration": {
        "um": 1.0,
        "µm": 1.0,
        "mm": 1000.0,
        "m": 1e6,
        "nm": 1e-3,
        "mol/l": 1e6,
    },
}


@dataclass(frozen=True)
class ConversionResult:
    ok: bool
    value: float
    unit: str
    diagnostic_code: str | None = None
    message: str | None = None
    raw: str | None = None


def convert(value: float, raw_unit: str, kind: str) -> ConversionResult:
    """Convert a scalar to the canonical unit for *kind*.

    Unconvertible / unknown units return ``ok=False`` with the original unit
    retained; callers must keep the value in place and surface the diagnostic.
    """
    if kind == "temperature":
        return _convert_temperature(value, raw_unit)
    table = _UNIT_FACTORS.get(kind)
    if table is None:
        return ConversionResult(False, value, raw_unit, "UNIT_KIND_UNKNOWN",
                                f"unknown quantity kind {kind!r}", raw_unit)
    key = raw_unit.strip().lower()
    factor = table.get(key)
    if factor is None:
        return ConversionResult(
            False, value, raw_unit, "UNIT_UNCONVERTIBLE",
            f"unit {raw_unit!r} cannot be converted to {CANONICAL[kind]}",
            raw_unit,
        )
    return ConversionResult(True, value * factor, CANONICAL[kind])


def _convert_temperature(value: float, raw_unit: str) -> ConversionResult:
    u = raw_unit.strip().lower()
    if u in ("k", "kelvin"):
        return ConversionResult(True, value, "K")
    if u in ("c", "°c", "degc", "degree_celsius"):
        return ConversionResult(True, value + 273.15, "K")
    return ConversionResult(
        False, value, raw_unit, "UNIT_UNCONVERTIBLE",
        f"temperature unit {raw_unit!r} cannot be converted to K", raw_unit,
    )


def convertable(value: float, raw_unit: str, kind: str) -> bool:
    return convert(value, raw_unit, kind).ok
