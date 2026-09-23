"""Quantities with units, coordinate bases and conversion diagnostics.

Import rule enforced here: the *original representation* is always preserved.
Normalisation is a derived operation; when a value cannot be converted it stays
in place (``norm is None``) with a diagnostic rather than being dropped or
replaced by a default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from .errors import Diagnostic, SemanticImpossibility

# ---------------------------------------------------------------------------
# Unit tables.  Factors convert the unit to the canonical (SI-ish) unit.
# ---------------------------------------------------------------------------

# energy -> joule
_ENERGY = {
    "J": 1.0,
    "mJ": 1e-3,
    "uJ": 1e-6,
    "µJ": 1e-6,
    "cal": 4.184,
    "mcal": 4.184e-3,
    "ucal": 4.184e-6,
    "µcal": 4.184e-6,
}
# power -> watt  (energy / second)
_POWER = {
    "W": 1.0,
    "mW": 1e-3,
    "uW": 1e-6,
    "µW": 1e-6,
    "J/s": 1.0,
    "cal/s": 4.184,
    "mcal/s": 4.184e-3,
    "ucal/s": 4.184e-6,
    "µcal/s": 4.184e-6,
}
# volume -> litre
_VOLUME = {
    "L": 1.0,
    "mL": 1e-3,
    "uL": 1e-6,
    "µL": 1e-6,
    "nL": 1e-9,
}
# time -> second
_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
         "min": 60.0, "minute": 60.0, "minutes": 60.0}
# concentration -> molar
_CONC = {
    "M": 1.0,
    "mol/L": 1.0,
    "mM": 1e-3,
    "um": 1e-6,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
}
# power rate -> watt/second
_POWER_RATE = {
    "W/s": 1.0,
    "mW/s": 1e-3,
    "uW/s": 1e-6,
    "µW/s": 1e-6,
    "uW/min": 1e-6 / 60,
    "µW/min": 1e-6 / 60,
}
# inverse concentration -> M^-1
_INV_CONC = {
    "M^-1": 1.0,
    "1/M": 1.0,
    "mM^-1": 1e3,
    "uM^-1": 1e6,
    "nM^-1": 1e9,
}

_DIMENSIONS = {
    "energy": (_ENERGY, "J"),
    "power": (_POWER, "W"),
    "volume": (_VOLUME, "L"),
    "time": (_TIME, "s"),
    "concentration": (_CONC, "M"),
    "inverse_concentration": (_INV_CONC, "M^-1"),
    "power_rate": (_POWER_RATE, "W/s"),
}

# Preferred display units per dimension (ITC conventions).
_PREFERRED = {
    "energy": "ucal",
    "power": "uW",
    "volume": "uL",
    "time": "s",
    "concentration": "uM",
    "inverse_concentration": "M^-1",
    "power_rate": "uW/s",
}

_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_TOKEN_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
                       r"\s*([^\s]+)?\s*$")


def canonical_unit(dimension: str) -> str:
    return _DIMENSIONS[dimension][1]


def preferred_unit(dimension: str) -> str:
    return _PREFERRED[dimension]


@dataclass
class Measurement:
    """A scalar measurement that remembers how it was written."""

    value: float | int | str | None
    unit: str
    original: str | None = None
    dimension: str | None = None
    basis: str | None = None        # coordinate / time basis label
    coord: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measurement":
        return cls(**d)


def _dimension_of(unit: str) -> str | None:
    unit = unit.replace("μ", "µ")
    for name, (table, _canon) in _DIMENSIONS.items():
        if unit in table:
            return name
    return None


def parse_measurement(raw: Any, location: str = "",
                      basis: str | None = None) -> tuple[Measurement | None, list[Diagnostic]]:
    """Parse a number+unit without losing the original text.

    Accepts numbers paired with an explicit unit string, or a single token such
    as ``"12.3 uW"``.  Returns the measurement and any diagnostics; a structurally
    missing value returns ``(None, [error diagnostic])``.
    """
    diagnostics: list[Diagnostic] = []
    if raw is None:
        return None, [Diagnostic("missing_value", "value is required", location)]
    if isinstance(raw, dict):
        value = raw.get("value")
        unit = raw.get("unit")
        original = raw.get("original")
        basis = raw.get("basis", basis)
        coord = raw.get("coord")
        if unit is None:
            diagnostics.append(Diagnostic("missing_unit", "unit is required",
                                          f"{location}.unit"))
            unit = ""
        if value is None:
            diagnostics.append(Diagnostic("missing_value", "value is required",
                                          f"{location}.value"))
        m = Measurement(value=value, unit=unit or "", original=original,
                        basis=basis, coord=coord)
        if unit and _dimension_of(unit) is None:
            diagnostics.append(Diagnostic("unknown_unit", f"unknown unit {unit!r}",
                                          f"{location}.unit", value=original or value))
        else:
            m.dimension = _dimension_of(unit)
        return m, diagnostics
    if isinstance(raw, (int, float)):
        diagnostics.append(Diagnostic("missing_unit",
                                      "bare number has no unit; value kept in place",
                                      location, "warning", raw))
        m = Measurement(value=raw, unit="", original=str(raw), basis=basis)
        return m, diagnostics
    text = str(raw)
    match = _TOKEN_RE.match(text)
    if not match:
        diagnostics.append(Diagnostic("unparseable_value",
                                      f"cannot parse measurement {text!r}",
                                      location, value=raw))
        m = Measurement(value=text, unit="", original=text, basis=basis)
        return m, diagnostics
    number_text, unit = match.group(1), match.group(2) or ""
    number: Any = float(number_text) if "." in number_text or "e" in number_text.lower() else int(number_text)
    m = Measurement(value=number, unit=unit, original=text, basis=basis)
    if unit and _dimension_of(unit) is None:
        diagnostics.append(Diagnostic("unknown_unit", f"unknown unit {unit!r}",
                                      f"{location}.unit", "error", text))
    else:
        m.dimension = _dimension_of(unit)
    return m, diagnostics


def convert(measurement: Measurement, target_unit: str,
            location: str = "") -> tuple[float | None, list[Diagnostic]]:
    """Convert to ``target_unit`` within the same dimension.

    Temperature is handled separately (offset, not factor).  An incompatible or
    unknown unit yields a diagnostic and ``None`` -- the original measurement is
    left untouched by the caller.
    """
    if measurement.unit in ("C", "°C", "degC", "K") and target_unit in ("C", "°C", "degC", "K"):
        try:
            v = float(measurement.value)
        except (TypeError, ValueError):
            return None, [Diagnostic("unconvertible_value",
                                     f"non-numeric temperature {measurement.value!r}",
                                     location, "error", measurement.original)]
        celsius = v if measurement.unit in ("C", "°C", "degC") else v - 273.15
        if target_unit in ("C", "°C", "degC"):
            return celsius, []
        return celsius + 273.15, []

    src_dim = _dimension_of(measurement.unit)
    tgt_dim = _dimension_of(target_unit)
    if src_dim is None:
        return None, [Diagnostic("unknown_unit",
                                 f"unknown source unit {measurement.unit!r}",
                                 f"{location}.unit", "error", measurement.original)]
    if tgt_dim is None:
        return None, [Diagnostic("unknown_unit",
                                 f"unknown target unit {target_unit!r}",
                                 location, "error")]
    if src_dim != tgt_dim:
        return None, [Diagnostic("incompatible_dimensions",
                                 f"cannot convert {measurement.unit!r} ({src_dim}) "
                                 f"to {target_unit!r} ({tgt_dim})",
                                 location, "error", measurement.original)]
    try:
        v = float(measurement.value)
    except (TypeError, ValueError):
        return None, [Diagnostic("unconvertible_value",
                                 f"non-numeric value {measurement.value!r}",
                                 location, "error", measurement.original)]
    table = _DIMENSIONS[src_dim][0]
    return v * table[measurement.unit] / table[target_unit], []


def require_number(measurement: Measurement | None, target_unit: str,
                   location: str) -> tuple[float | None, list[Diagnostic]]:
    if measurement is None:
        return None, [Diagnostic("missing_value", "measurement is required", location)]
    return convert(measurement, target_unit, location)


def is_unit_of(unit: str, dimension: str) -> bool:
    return _dimension_of(unit) == dimension
