"""Quantities, units and coordinate bases.

Raw representations are preserved on import; normalisation happens only in the
derived layer. Values that cannot be converted stay at their original location
with a diagnostic rather than being dropped or silently defaulted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import SemanticUnresolved

# multipliers to canonical (SI-like internal) units
TO_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "min": 60.0, "m": 60.0}
TO_LITRES = {"l": 1.0, "ml": 1e-3, "ul": 1e-6, "µl": 1e-6}
TO_WATT = {"w": 1.0, "uw": 1e-6, "µw": 1e-6, "mw": 1e-3, "cal/s": 4.184, "ucal/s": 4.184e-6}
TO_MOLAR = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "mol/l": 1.0}
TO_KELVIN_DELTA = {"k": 1.0, "c": 1.0, "°c": 1.0}
TO_JOULE = {
    "j": 1.0,
    "mj": 1e-3,
    "uj": 1e-6,
    "µj": 1e-6,
    "cal": 4.184,
    "kcal": 4184.0,
}
TO_JOULE_PER_MOLE = {
    "j/mol": 1.0,
    "kj/mol": 1e3,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
}

_ALIASES = {
    "time": TO_SECONDS,
    "volume": TO_LITRES,
    "power": TO_WATT,
    "concentration": TO_MOLAR,
    "temperature_delta": TO_KELVIN_DELTA,
    "energy": TO_JOULE,
    "molar_energy": TO_JOULE_PER_MOLE,
}

CANONICAL_UNIT = {
    "time": "s",
    "volume": "L",
    "power": "W",
    "concentration": "M",
    "temperature": "K",
    "temperature_delta": "K",
    "energy": "J",
    "molar_energy": "J/mol",
}


@dataclass(frozen=True)
class Quant:
    """A scalar carrying its raw text, unit and normalised value."""

    value: float
    raw: str
    unit: str
    canonical_unit: str
    dimension: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw": self.raw,
            "unit": self.unit,
            "canonical_unit": self.canonical_unit,
            "dimension": self.dimension,
        }


def normalize(raw: Any, unit: str | None, dimension: str, *, position: str) -> Quant:
    raw_text = "" if raw is None else str(raw).strip()
    unit_text = (unit or "").strip().replace("μ", "µ")
    if raw_text == "":
        raise SemanticUnresolved(
            "missing value; no default is substituted",
            diagnostics=[{"position": position, "reason": "EMPTY_VALUE", "raw": raw_text}],
        )
    try:
        number = float(raw_text)
    except (TypeError, ValueError):
        raise SemanticUnresolved(
            f"non-numeric value {raw_text!r} kept at {position}",
            diagnostics=[{"position": position, "reason": "NON_NUMERIC", "raw": raw_text}],
        )
    if not math.isfinite(number):
        raise SemanticUnresolved(
            f"non-finite value {raw_text!r} kept at {position}",
            diagnostics=[{"position": position, "reason": "NON_FINITE", "raw": raw_text}],
        )

    if dimension == "temperature":
        unit_l = unit_text.lower()
        if unit_l in ("c", "°c"):
            kelvin = number + 273.15
        elif unit_l in ("k", ""):
            kelvin = number
        else:
            raise SemanticUnresolved(
                f"unknown temperature unit {unit_text!r}",
                diagnostics=[{"position": position, "reason": "UNKNOWN_UNIT", "raw": raw_text, "unit": unit_text}],
            )
        return Quant(kelvin, raw_text, unit_text or "K", "K", "temperature")

    table = _ALIASES.get(dimension)
    if table is None:
        raise SemanticUnresolved(
            f"unknown dimension {dimension!r}",
            diagnostics=[{"position": position, "reason": "UNKNOWN_DIMENSION"}],
        )
    factor = table.get(unit_l := unit_text.lower())
    if factor is None:
        raise SemanticUnresolved(
            f"cannot convert {raw_text} {unit_text} ({dimension}); value left in place",
            diagnostics=[
                {
                    "position": position,
                    "reason": "CONVERSION_FAILED",
                    "raw": raw_text,
                    "unit": unit_text,
                    "dimension": dimension,
                }
            ],
        )
    return Quant(number * factor, raw_text, unit_text, CANONICAL_UNIT[dimension], dimension)
