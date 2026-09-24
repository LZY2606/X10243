"""Units and coordinate bases.

All domain quantities carry their unit. The *raw representation* is preserved
on import; the derived layers normalise with the factors in this module.

Canonical units (used only in derived layers):

* time          : s
* volume        : uL
* concentration : mM
* power         : ucal/s
* energy        : ucal
* temperature   : degC
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, SemanticImpasse

CAL4184 = 4.184  # 1 cal = 4.184 J

_TO_CANON: dict[str, dict[str, float]] = {
    "time": {
        "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
        "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3,
        "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    },
    "volume": {
        "ul": 1.0, "µl": 1.0, "μl": 1.0, "microliter": 1.0, "microliters": 1.0, "µl": 1.0,
        "ml": 1e3, "milliliter": 1e3, "milliliters": 1e3,
        "l": 1e6, "liter": 1e6, "liters": 1e6,
    },
    "concentration": {
        "mm": 1.0, "mmol/l": 1.0, "mmolar": 1.0,
        "m": 1e3, "mol/l": 1e3, "molar": 1e3,
        "um": 1e-3, "µm": 1e-3, "μm": 1e-3, "umol/l": 1e-3, "µmol/l": 1e-3,
        "nm": 1e-6, "nmol/l": 1e-6,
    },
    "power": {
        "ucal/s": 1.0, "µcal/s": 1.0, "μcal/s": 1.0, "microcal/s": 1.0,
        "mcal/s": 1e3, "millical/s": 1e3,
        "cal/s": 1e6,
        "uw": 1.0 / CAL4184, "µw": 1.0 / CAL4184, "μw": 1.0 / CAL4184, "microwatt": 1.0 / CAL4184,
        "mw": 1e3 / CAL4184, "milliwatt": 1e3 / CAL4184,
        "w": 1e6 / CAL4184, "j/s": 1e6 / CAL4184,
    },
    "energy": {
        "ucal": 1.0, "µcal": 1.0, "μcal": 1.0, "microcal": 1.0,
        "mcal": 1e3, "millical": 1e3,
        "cal": 1e6, "calorie": 1e6,
        "kcal": 1e9, "kilocal": 1e9,
        "uj": 1.0 / CAL4184, "µj": 1.0 / CAL4184, "μj": 1.0 / CAL4184, "microjoule": 1.0 / CAL4184,
        "mj": 1e3 / CAL4184, "millijoule": 1e3 / CAL4184,
        "j": 1e6 / CAL4184, "joule": 1e6 / CAL4184,
        "kj": 1e9 / CAL4184, "kilojoule": 1e9 / CAL4184,
    },
}

CANONICAL_UNIT = {
    "time": "s", "volume": "uL", "concentration": "mM",
    "power": "ucal/s", "energy": "ucal", "temperature": "degC",
}

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def normalise_unit(unit: str) -> str:
    u = unit.strip().lower()
    u = u.replace("μ", "μ").replace("µ", "µ")
    return u


@dataclass(frozen=True)
class Quantity:
    raw: str
    value: float | None
    unit: str | None
    dimension: str
    canonical: float | None = None
    canonical_unit: str | None = None
    convertible: bool = True
    diagnostic: Diagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"raw": self.raw, "unit": self.unit, "dimension": self.dimension,
               "canonical_unit": self.canonical_unit, "convertible": self.convertible}
        if self.value is not None:
            out["value"] = self.value
        if self.canonical is not None:
            out["canonical"] = self.canonical
        if self.diagnostic is not None:
            out["diagnostic"] = self.diagnostic.to_dict()
        return out


def _split(text: str) -> tuple[float | None, str | None]:
    token = text.strip()
    m = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", token)
    if not m:
        return None, None
    return float(m.group(1)), m.group(2).strip() or None


def parse_quantity(text: Any, dimension: str, *, default_unit: str | None = None,
                   file: str | None = None, row: int | None = None,
                   column: str | None = None) -> Quantity:
    """Parse ``"1.42 mM"`` preserving the original representation.

    Unconvertible / non-numeric values are returned in place with a diagnostic
    (``convertible=False``); they are never dropped for plotting.
    """
    raw = "" if text is None else str(text)
    cu = CANONICAL_UNIT[dimension]
    if dimension == "temperature":
        return _parse_temperature(raw, file, row, column)
    value, unit = _split(raw)
    if value is None:
        return Quantity(raw, None, unit, dimension, None, cu, False,
                        Diagnostic("not_a_number", f"value {raw!r} is not numeric",
                                   file, row, column))
    unit = unit or default_unit
    if unit is None:
        return Quantity(raw, value, None, dimension, None, cu, False,
                        Diagnostic("missing_unit", "no unit supplied and no default",
                                   file, row, column))
    factor = _TO_CANON[dimension].get(normalise_unit(unit))
    if factor is None:
        return Quantity(raw, value, unit, dimension, None, cu, False,
                        Diagnostic("unknown_unit", f"unit {unit!r} is not convertible for {dimension}",
                                   file, row, column, {"unit": unit}))
    return Quantity(raw, value, unit, dimension, value * factor, cu, True)


def _parse_temperature(raw: str, file: str | None, row: int | None,
                       column: str | None) -> Quantity:
    value, unit = _split(raw.replace("°C", "C").replace("℃", "C")
                              .replace("°c", "C").replace("°F", "F").replace("℉", "F"))
    if value is None:
        return Quantity(raw, None, None, "temperature", None, "degC", False,
                        Diagnostic("not_a_number", f"temperature {raw!r} is not numeric",
                                   file, row, column))
    unit = unit or "C"
    key = normalise_unit(unit)
    if key in ("c", "degc", "celsius", "°c"):
        canon = value
    elif key in ("k", "kelvin"):
        canon = value - 273.15
    elif key in ("f", "degf", "fahrenheit"):
        canon = (value - 32.0) * 5.0 / 9.0
    else:
        return Quantity(raw, value, unit, "temperature", None, "degC", False,
                        Diagnostic("unknown_unit", f"temperature unit {unit!r} not convertible",
                                   file, row, column, {"unit": unit}))
    return Quantity(raw, value, unit, "temperature", canon, "degC", True)


def require(q: Quantity) -> float:
    """Raise :class:`SemanticImpasse` if ``q`` cannot be normalised."""
    if not q.convertible or q.canonical is None:
        raise SemanticImpasse(
            f"unconvertible {q.dimension} value {q.raw!r}",
            [q.diagnostic] if q.diagnostic else None)
    return q.canonical


def convert(value: float, unit: str, dimension: str) -> float:
    factor = _TO_CANON[dimension].get(normalise_unit(unit))
    if factor is None:
        raise SemanticImpasse(f"unit {unit!r} not convertible for {dimension}",
                              [Diagnostic("unknown_unit", "unconvertible unit",
                                          column=dimension, detail={"unit": unit})])
    return value * factor
