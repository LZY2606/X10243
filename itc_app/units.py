"""Unit-aware quantities.

Every imported domain value keeps its *raw representation* (string + declared
unit). Normalisation happens only in the derivation layer. A value that cannot
be converted stays in place and carries a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RULES_VERSION = "ITC-RULES-2026.09.1"

# Conversion factors into the canonical internal unit of each dimension.
_TO_CANONICAL: dict[tuple[str, str], float] = {
    ("power", "uW"): 1.0,
    ("power", "uJ/s"): 1.0,
    ("power", "mW"): 1000.0,
    ("power", "W"): 1.0e6,
    ("energy", "uJ"): 1.0,
    ("energy", "mJ"): 1000.0,
    ("energy", "J"): 1.0e6,
    ("energy", "cal"): 4.184,
    ("energy", "kcal"): 4184.0,
    ("volume", "uL"): 1.0,
    ("volume", "mL"): 1000.0,
    ("volume", "L"): 1.0e6,
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "min"): 60.0,
    ("time", "h"): 3600.0,
    ("concentration", "mM"): 1.0,
    ("concentration", "uM"): 1.0e-3,
    ("concentration", "nM"): 1.0e-6,
    ("concentration", "M"): 1000.0,
    ("temperature", "K"): 1.0,
    ("temperature", "C"): 1.0,
    ("temperature", "celsius"): 1.0,
    ("molar_energy", "kcal/mol"): 1.0,
    ("molar_energy", "cal/mol"): 1.0e-3,
    ("molar_energy", "kJ/mol"): 0.239006,
    ("molar_energy", "J/mol"): 0.239006e-3,
}

CANONICAL_UNIT = {
    "power": "uW",
    "energy": "uJ",
    "volume": "uL",
    "time": "s",
    "concentration": "mM",
    "temperature": "K",
    "molar_energy": "kcal/mol",
}


@dataclass
class Diagnostic:
    """A positioned diagnostic. Never silently replaced by a default."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    position: str | None = None
    value: Any = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "position": self.position,
            "value": self.value,
        }


@dataclass
class Quantity:
    """Raw value as written, declared unit, and (lazily) canonical value."""

    raw: str
    unit: str
    dimension: str
    value: float | None = None
    canonical: float | None = None
    convertible: bool = True
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "dimension": self.dimension,
            "value": self.value,
            "canonical": self.canonical,
            "convertible": self.convertible,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def parse_quantity(raw: Any, unit: str, dimension: str, position: str | None = None) -> Quantity:
    """Parse a scalar while preserving its raw representation.

    Unparseable or unconvertible inputs return a Quantity kept in place with a
    diagnostic; callers must surface it, never substitute a default.
    """
    raw_str = "" if raw is None else str(raw).strip()
    q = Quantity(raw=raw_str, unit=(unit or "").strip(), dimension=dimension)
    try:
        q.value = float(raw_str)
    except (TypeError, ValueError):
        q.convertible = False
        q.canonical = None
        q.diagnostics.append(
            Diagnostic(
                "E_NUM_PARSE",
                "error",
                f"cannot parse {dimension!r} value {raw_str!r}",
                position,
                raw_str,
            )
        )
        return q
    key = (dimension, q.unit)
    if key not in _TO_CANONICAL:
        q.convertible = False
        q.diagnostics.append(
            Diagnostic(
                "E_UNIT_UNKNOWN",
                "error",
                f"unit {q.unit!r} is not convertible on dimension {dimension!r}",
                position,
                q.unit,
            )
        )
        return q
    factor = _TO_CANONICAL[key]
    if dimension == "temperature" and q.unit in ("C", "celsius"):
        q.canonical = q.value + 273.15
    else:
        q.canonical = q.value * factor
    return q


def to_canonical(raw: Any, unit: str, dimension: str, position: str | None = None) -> tuple[float | None, list[Diagnostic]]:
    q = parse_quantity(raw, unit, dimension, position)
    return q.canonical, q.diagnostics


def convert_from_canonical(value: float, dimension: str) -> tuple[float, str]:
    unit = CANONICAL_UNIT[dimension]
    key = (dimension, unit)
    factor = _TO_CANONICAL[key]
    if dimension == "temperature":
        return value, unit  # K offset handled at parse time only
    return value / factor, unit
