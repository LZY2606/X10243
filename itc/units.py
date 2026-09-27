"""Unit parsing and normalisation.

Rule: imported values keep their raw representation. Normalisation only
happens in the derived layer. Values that cannot be converted stay in
place with a diagnostic; they are never dropped or replaced by defaults.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .errors import Diagnostic

# canonical kind -> {unit: multiplier into canonical unit}
UNIT_TABLE = {
    "volume_uL": {"uL": 1.0, "µL": 1.0, "ul": 1.0, "mL": 1000.0, "ml": 1000.0},
    "conc_mM": {"mM": 1.0, "uM": 1e-3, "µM": 1e-3, "nM": 1e-6, "M": 1000.0},
    "temperature_K": {"K": 1.0},
    "power_uJ_s": {"uJ/s": 1.0, "µJ/s": 1.0, "ucal/s": 4.184, "µcal/s": 4.184},
    "time_s": {"s": 1.0, "sec": 1.0, "min": 60.0},
    "enthalpy_kJ_mol": {"kJ/mol": 1.0, "kcal/mol": 4.184},
    "heat_uJ": {"uJ": 1.0, "µJ": 1.0, "ucal": 4.184, "µcal": 4.184},
}

_UNIT_TO_KIND = {}
for _kind, _table in UNIT_TABLE.items():
    for _unit in _table:
        _UNIT_TO_KIND[_unit] = _kind

_QTY_RE = re.compile(
    r"^\s*(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>[^\s]+))?\s*$"
)


@dataclass
class Quantity:
    """A parsed quantity. Raw text is always preserved."""

    raw: str
    position: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    canonical_kind: Optional[str] = None
    canonical_value: Optional[float] = None
    canonical_unit: Optional[str] = None
    diagnostic: Optional[Diagnostic] = None

    @property
    def ok(self) -> bool:
        return self.diagnostic is None and self.canonical_value is not None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "position": self.position,
            "value": self.value,
            "unit": self.unit,
            "canonical_kind": self.canonical_kind,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }


def parse_quantity(raw, position: Optional[str] = None) -> Quantity:
    """Parse '12.5 uL' style input. Never raises for bad values; the
    unconvertible value stays in place with a diagnostic."""
    q = Quantity(raw=raw, position=position)
    if not isinstance(raw, str):
        q.diagnostic = Diagnostic(
            code="value.not_text",
            message=f"expected text quantity, got {type(raw).__name__}",
            position=position, severity="error")
        return q
    m = _QTY_RE.match(raw)
    if not m:
        q.diagnostic = Diagnostic(
            code="value.unparseable",
            message=f"cannot parse quantity: {raw!r}",
            position=position, severity="error")
        return q
    q.value = float(m.group("num"))
    unit = m.group("unit")
    q.unit = unit
    if unit is None:
        q.diagnostic = Diagnostic(
            code="unit.missing",
            message=f"quantity {raw!r} has no unit; left unnormalised",
            position=position, severity="error")
        return q
    kind = _UNIT_TO_KIND.get(unit)
    if kind is None:
        q.diagnostic = Diagnostic(
            code="unit.unconvertible",
            message=f"no conversion rule for unit {unit!r}; value kept raw",
            position=position, severity="error")
        return q
    q.canonical_kind = kind
    q.canonical_value = q.value * UNIT_TABLE[kind][unit]
    q.canonical_unit = {
        "volume_uL": "uL", "conc_mM": "mM", "temperature_K": "K",
        "power_uJ_s": "uJ/s", "time_s": "s",
        "enthalpy_kJ_mol": "kJ/mol", "heat_uJ": "uJ",
    }[kind]
    return q
