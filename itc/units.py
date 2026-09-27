"""Quantity handling: every domain value keeps its unit/coordinate basis.

The import layer preserves the *raw representation*; the derivation layer is
the only place that normalises to SI-ish working units. Values that cannot be
converted stay in place (``value is None``) with a structured diagnostic -
nothing is dropped merely to make plotting convenient.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .errors import SemanticUnsolvable

# dimension -> {canonical unit: multiplicative factor to the normalised unit}
# Temperature needs affine conversion and is handled separately.
_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
             "min": 60.0, "minute": 60.0, "minutes": 60.0,
             "ms": 1e-3, "h": 3600.0, "hr": 3600.0},
    "volume": {"uL": 1e-3, "µL": 1e-3, "ul": 1e-3, "mL": 1.0, "ml": 1.0,
               "L": 1e3},
    "concentration": {"mM": 1.0, "mmol/L": 1.0, "um": 1e-3, "µM": 1e-3,
                      "uM": 1e-3, "M": 1e3, "mol/L": 1e3, "nM": 1e-6},
    "power": {"ucal/s": 1.0, "µcal/s": 1.0, "uCal/s": 1.0, "mcal/s": 1e3.0,
              "cal/s": 1e6},
    "energy": {"ucal": 1.0, "µcal": 1.0, "mcal": 1e3, "cal": 1e6,
               "kcal": 1e9, "J": 1e6 * 4.184, "mJ": 1e3 * 4.184},
    "molar_energy": {"kcal/mol": 1.0, "cal/mol": 1e-3,
                     "kJ/mol": 1e3 / 4.184, "J/mol": 1.0 / 4.184},
}

_TEMPERATURE_UNITS = {"C": "celsius", "°C": "celsius", "degC": "celsius",
                      "K": "kelvin", "F": "fahrenheit", "°F": "fahrenheit"}

_NORMALISED = {
    "time": "s", "volume": "mL", "concentration": "mM", "power": "ucal/s",
    "energy": "ucal", "molar_energy": "kcal/mol",
}

_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)\s*$")


def normalised_unit(dimension: str) -> str:
    return _NORMALISED[dimension]


def make_quantity(raw: Any, dimension: str, *, basis: str | None = None,
                  where: str | None = None) -> dict:
    """Parse *raw* into a quantity dict, never discarding the raw text.

    Unconvertible / unparseable input yields ``value=None`` plus a diagnostic
    instead of raising, so ingestion keeps the material intact.
    """
    raw_text = str(raw)
    qty: dict[str, Any] = {
        "dimension": dimension,
        "raw": raw_text,
        "unit": None,
        "value": None,
        "normalised_value": None,
        "normalised_unit": None,
        "basis": basis,
        "diagnostics": [],
    }
    match = _NUMBER_RE.match(raw_text)
    if not match:
        qty["diagnostics"].append({
            "kind": "unparseable_number",
            "where": where,
            "raw": raw_text,
            "message": f"cannot read number from {raw_text!r}",
        })
        return qty
    number = float(match.group(1))
    unit = match.group(2) or ""
    qty["value"] = number
    if math.isnan(number) or math.isinf(number):
        qty["value"] = None
        qty["diagnostics"].append({
            "kind": "non_finite_number", "where": where, "raw": raw_text})
        return qty

    if dimension == "temperature":
        scale = _TEMPERATURE_UNITS.get(unit)
        if scale is None:
            qty["unit"] = unit or None
            qty["diagnostics"].append({
                "kind": "unconvertible_unit", "where": where, "raw": raw_text,
                "unit": unit,
                "message": f"unknown temperature unit {unit!r}",
            })
            return qty
        qty["unit"] = unit
        if scale == "celsius":
            kelvin = number + 273.15
        elif scale == "fahrenheit":
            kelvin = (number - 32.0) * 5.0 / 9.0 + 273.15
        else:
            kelvin = number
        qty["normalised_value"] = round(kelvin, 6)
        qty["normalised_unit"] = "K"
        return qty

    table = _FACTORS.get(dimension)
    if table is None:
        raise SemanticUnsolvable(f"unknown dimension {dimension!r}")
    factor = table.get(unit)
    if factor is None:
        qty["unit"] = unit or None
        qty["diagnostics"].append({
            "kind": "unconvertible_unit", "where": where, "raw": raw_text,
            "unit": unit,
            "message": f"unit {unit!r} not convertible for {dimension!r}",
        })
        return qty
    qty["unit"] = unit
    qty["normalised_value"] = number * factor
    qty["normalised_unit"] = _NORMALISED[dimension]
    return qty


def require(qty: dict, *, where: str | None = None) -> float:
    """Return the normalised scalar or raise a located semantic error."""
    if qty.get("normalised_value") is None:
        diags = qty.get("diagnostics") or []
        raise SemanticUnsolvable(
            f"no usable value for quantity at {where or qty.get('raw')!r}",
            diagnostics=diags)
    return float(qty["normalised_value"])


def as_display(qty: dict) -> str:
    if qty.get("normalised_value") is not None:
        return f"{qty['normalised_value']:g} {qty['normalised_unit']}"
    return f"{qty['raw']} (raw, not normalised)"
