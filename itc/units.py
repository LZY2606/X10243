"""Quantities with units / coordinate bases.

Raw representations are preserved on import; only the derived layer performs
normalization. A value that cannot be converted stays in place with a
diagnostic instead of being dropped or silently defaulted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic

# dimension -> {unit_label: multiplicative factor to canonical unit}
UNITS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
             "ms": 1e-3, "msec": 1e-3, "min": 60.0, "minute": 60.0},
    "power": {"uW": 1.0, "µW": 1.0, "μW": 1.0, "mW": 1e3, "W": 1e6},
    "volume": {"uL": 1.0, "µL": 1.0, "μL": 1.0, "mL": 1e3, "L": 1e6},
    "concentration": {"uM": 1.0, "µM": 1.0, "μM": 1.0, "mM": 1e3,
                      "M": 1e6, "nM": 1e-3},
    "temperature": {"C": 1.0, "°C": 1.0, "K": 1.0, "kelvin": 1.0},
    "energy": {"uJ": 1.0, "µJ": 1.0, "μJ": 1.0, "mJ": 1e3, "J": 1e6,
               "cal": 4.184e6, "kcal": 4.184e9},
    "enthalpy": {"kcal/mol": 1.0, "kJ/mol": 4.184 / 4.184 * 0.0,
                 "cal/mol": 1e-3, "J/mol": 0.001 / 4.184},
}
# enthalpy canonical unit: kcal/mol
UNITS["enthalpy"] = {"kcal/mol": 1.0, "kcal": 1.0, "cal/mol": 1e-3,
                     "kJ/mol": 1.0 / 4.184, "J/mol": 1e-3 / 4.184}
# affinity canonical unit: 1/uM = 1e6 1/M
UNITS["affinity"] = {"1/uM": 1.0, "1/µM": 1.0, "1/μM": 1.0,
                     "M^-1": 1e-6, "1/M": 1e-6, "L/mol": 1e-6}
UNITS["molar_heat"] = {"kcal/mol": 1.0, "kJ/mol": 1.0 / 4.184,
                       "cal/mol": 1e-3, "J/mol": 1e-3 / 4.184}

CANONICAL = {
    "time": "s", "power": "uW", "volume": "uL", "concentration": "uM",
    "temperature": "degC", "energy": "uJ", "enthalpy": "kcal/mol",
    "affinity": "1/uM", "molar_heat": "kcal/mol",
}


@dataclass
class Quantity:
    """A value carrying its original textual representation.

    ``raw`` is exactly what the source file provided; ``value``/``unit`` are
    the normalized form and are None when conversion is impossible.
    """

    raw: Any
    value: float | None = None
    unit: str | None = None
    dimension: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None and not any(
            d.severity == "error" for d in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"raw": self.raw, "value": self.value,
                               "unit": self.unit, "dimension": self.dimension}
        if self.diagnostics:
            out["diagnostics"] = [d.to_dict() for d in self.diagnostics]
        return out


def parse_quantity(spec: Any, dimension: str, location: str = "") -> Quantity:
    """Parse ``{"value":.., "unit":..}`` / number / "12.3 uM" into a Quantity."""
    if isinstance(spec, Quantity):
        return spec
    q = Quantity(raw=spec, dimension=dimension)
    raw_value: Any = spec
    raw_unit: str | None = None
    if isinstance(spec, dict):
        raw_value = spec.get("value")
        raw_unit = spec.get("unit")
    elif isinstance(spec, str):
        text = spec.strip()
        # split trailing unit token, e.g. "298.15 K", "10.5 µL"
        for i in range(len(text) - 1, -1, -1):
            if text[i].isalpha() or text[i] in "°µμ^/-":
                continue
            raw_unit = text[i + 1:].strip() or None
            text = text[: i + 1].strip()
            break
        raw_value = text
    if raw_unit is None:
        raw_unit = CANONICAL[dimension]
    try:
        num = float(raw_value)
    except (TypeError, ValueError):
        q.diagnostics.append(Diagnostic(
            "unparseable_number", f"cannot read number from {raw_value!r}",
            location, "error"))
        return q
    if math.isnan(num) or math.isinf(num):
        q.diagnostics.append(Diagnostic(
            "non_finite", f"value {raw_value!r} is not finite", location, "error"))
        return q
    table = UNITS.get(dimension)
    if table is None:
        q.diagnostics.append(Diagnostic(
            "unknown_dimension", f"no unit table for {dimension!r}", location, "error"))
        return q
    factor = table.get(raw_unit)
    if factor is None:
        # value kept at original position; never dropped
        q.diagnostics.append(Diagnostic(
            "unconvertible_unit",
            f"unit {raw_unit!r} cannot be converted for dimension {dimension!r}",
            location, "error"))
        return q
    q.value = num * factor if dimension != "temperature" else _convert_temp(num, raw_unit)
    q.unit = CANONICAL[dimension]
    return q


def _convert_temp(num: float, unit: str) -> float:
    if unit in ("K", "kelvin"):
        return num - 273.15
    return num


def convert(value: float, unit: str, dimension: str) -> float:
    factor = UNITS[dimension][unit]
    if dimension == "temperature":
        return _convert_temp(value, unit)
    return value * factor


def require(q: Quantity, location: str = "") -> float:
    if not q.ok:
        from .errors import SemanticUnresolvable

        d = next((d for d in q.diagnostics if d.severity == "error"), None)
        raise SemanticUnresolvable(
            d.message if d else "quantity unavailable", location=location)
    return q.value
