"""Unit parsing and normalisation.

The raw representation is always preserved on the material; only the derived
layer normalises. Values that cannot be converted stay in place with a
diagnostic (they are never dropped to keep a plot renderable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic

# internal canonical units
TIME_S = "s"
TEMP_K = "K"
VOLUME_L = "L"
CONC_M = "M"
POWER_UCA = "ucal/s"
ENERGY_UCA = "ucal"
ENERGY_UJ = "uJ"

_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
         "min": 60.0, "minute": 60.0, "minutes": 60.0}
_VOLUME = {"l": 1.0, "liter": 1.0, "liters": 1.0, "ml": 1e-3,
           "ul": 1e-6, "µl": 1e-6, "μl": 1e-6}
_CONC = {"m": 1.0, "mol/l": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6,
         "μm": 1e-6, "nm": 1e-9}
_POWER = {"ucal/s": 1.0, "µcal/s": 1.0, "μcal/s": 1.0,
          "cal/s": 1e6, "uj/s": 1.0 / 4.184, "µj/s": 1.0 / 4.184,
          "μj/s": 1.0 / 4.184}
_ENERGY = {"ucal": 1.0, "µcal": 1.0, "μcal": 1.0, "cal": 1e6,
           "uj": 1.0 / 4.184, "µj": 1.0 / 4.184, "μj": 1.0 / 4.184,
           "j": 1.0 / 4.184e6}

_TABLES = {
    "time": (_TIME, TIME_S),
    "volume": (_VOLUME, VOLUME_L),
    "concentration": (_CONC, CONC_M),
    "power": (_POWER, POWER_UCA),
    "energy": (_ENERGY, ENERGY_UCA),
}


def _norm_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    return str(unit).strip().lower().replace(" ", "")


@dataclass
class NumericWithUnit:
    value: float
    unit: str
    canonical_unit: str
    raw: str | None = None


def convert_scalar(value: Any, unit: str | None, kind: str, position: str):
    """Return (normalised_value, canonical_unit, diagnostics).

    A non-convertible value returns (None, None, [Diagnostic]) and the caller
    keeps the original representation rather than substituting a default.
    """
    table, canonical = _TABLES[kind]
    raw_unit = unit
    key = _norm_unit(unit)
    try:
        num = float(value)
    except (TypeError, ValueError):
        diag = Diagnostic("error", "NON_NUMERIC_VALUE",
                          f"value {value!r} is not numeric",
                          position=position, value=value)
        return None, None, [diag.to_dict()]
    if key not in table:
        diag = Diagnostic("error", "UNCONVERTIBLE_UNIT",
                          f"unit {unit!r} cannot be converted to {canonical}",
                          position=position, value={"value": value, "unit": unit})
        return None, None, [diag.to_dict()]
    return num * table[key], canonical, []


def convert_temperature(value: Any, unit: str | None, position: str):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, None, [Diagnostic("error", "NON_NUMERIC_VALUE",
                                       f"temperature {value!r} is not numeric",
                                       position=position, value=value).to_dict()]
    key = _norm_unit(unit)
    if key in ("k", "kelvin"):
        return num, TEMP_K, []
    if key in ("c", "°c", "degc", "celsius"):
        return num + 273.15, TEMP_K, []
    if key in ("f", "°f", "degf", "fahrenheit"):
        return (num - 32.0) * 5.0 / 9.0 + 273.15, TEMP_K, []
    return None, None, [Diagnostic("error", "UNCONVERTIBLE_UNIT",
                                   f"temperature unit {unit!r} not supported",
                                   position=position,
                                   value={"value": value, "unit": unit}).to_dict()]


def ucal_to_uj(ucal: float) -> float:
    return ucal * 4.184
