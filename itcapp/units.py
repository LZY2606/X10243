"""Unit handling for ITC domain quantities.

Raw representations are preserved on import; normalization to canonical
units happens only in the derived layer. Values that cannot be converted
stay in place and carry a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# canonical (SI-ish) unit per quantity kind
CANONICAL = {
    "power": "uJ/s",
    "time": "s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "heat": "uJ",
    "molar_heat": "kcal/mol",
}

# (unit string, quantity kind) -> factor to canonical unit
_FACTORS = {
    ("uJ/s", "power"): 1.0,
    ("uW", "power"): 1.0,
    ("mJ/s", "power"): 1.0e3,
    ("mW", "power"): 1.0e3,
    ("ucal/s", "power"): 4.184,
    ("J/s", "power"): 1.0e6,
    ("W", "power"): 1.0e6,
    ("s", "time"): 1.0,
    ("sec", "time"): 1.0,
    ("min", "time"): 60.0,
    ("uL", "volume"): 1.0,
    ("ul", "volume"): 1.0,
    ("mL", "volume"): 1.0e3,
    ("L", "volume"): 1.0e6,
    ("mM", "concentration"): 1.0,
    ("uM", "concentration"): 1.0e-3,
    ("M", "concentration"): 1.0e3,
    ("K", "temperature"): 1.0,
    ("C", "temperature"): None,  # affine, handled specially
    ("degC", "temperature"): None,
    ("uJ", "heat"): 1.0,
    ("mJ", "heat"): 1.0e3,
    ("J", "heat"): 1.0e6,
    ("ucal", "heat"): 4.184,
    ("cal", "heat"): 4184.0,
    ("kcal/mol", "molar_heat"): 1.0,
    ("kJ/mol", "molar_heat"): 4.184,
    ("J/mol", "molar_heat"): 4.184e-3,
    ("cal/mol", "molar_heat"): 1.0e-3,
}


@dataclass
class Quantity:
    """A domain quantity: raw token preserved, canonical value derived."""

    raw: str
    unit: str
    kind: str
    value: Optional[float] = None  # canonical-unit value, None if unconvertible
    diagnostic: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "kind": self.kind,
            "value": self.value,
            "canonical_unit": CANONICAL.get(self.kind),
            "diagnostic": self.diagnostic,
        }


def convert(raw: str, unit: str, kind: str, location: str = "") -> Quantity:
    """Convert a raw numeric token with a unit into canonical units.

    Never raises for unit problems: unconvertible values keep their raw
    representation and carry a UNIT_UNKNOWN diagnostic.
    """
    q = Quantity(raw=raw, unit=unit, kind=kind)
    try:
        x = float(raw)
    except (TypeError, ValueError):
        q.diagnostic = {
            "code": "VALUE_NOT_NUMERIC",
            "severity": "error",
            "location": location,
            "message": f"value {raw!r} is not numeric; raw kept in place",
        }
        return q
    key = (unit, kind)
    if key not in _FACTORS:
        q.diagnostic = {
            "code": "UNIT_UNKNOWN",
            "severity": "error",
            "location": location,
            "message": (
                f"unit {unit!r} unknown for {kind}; raw value kept in place, "
                "not converted"
            ),
        }
        return q
    factor = _FACTORS[key]
    if factor is None:  # affine temperature C -> K
        q.value = x + 273.15
    else:
        q.value = x * factor
    return q
