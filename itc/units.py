"""Quantities with preserved raw representation and canonical SI units.

A :class:`Quantity` keeps the exact token the laboratory supplied
(``raw``) plus its unit string.  Conversion happens only at the derived
layer; values that cannot be converted stay in place with a diagnostic
instead of being silently dropped.

Canonical units used after normalization:

    power       W            (uW -> 1e-6, cal/s -> 4.184)
    energy      J            (uJ -> 1e-6, mcal -> 4.184e-3, cal -> 4.184)
    heat/mole   J/mol        (uJ/mol, cal/mol=4.184, kcal/mol=4184.0)
    volume      L            (uL -> 1e-6, mL -> 1e-3)
    concentration mol/L = M  (mM -> 1e-3, uM -> 1e-6, nM -> 1e-9)
    temperature K            (degC offset 273.15; K identity)
    time        s            (ms -> 1e-3, min -> 60.0, hr -> 3600.0)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, SemanticImpasse

_TOKEN_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^0-9\s].*?)?\s*$")

# factor: multiply numeric value by factor to reach the canonical unit.
_FACTORS: dict[tuple[str, str], float] = {
    # power -> W
    ("power", "W"): 1.0,
    ("power", "mW"): 1e-3,
    ("power", "uW"): 1e-6,
    ("power", "µW"): 1e-6,
    ("power", "cal/s"): 4.184,
    # energy -> J
    ("energy", "J"): 1.0,
    ("energy", "mJ"): 1e-3,
    ("energy", "uJ"): 1e-6,
    ("energy", "µJ"): 1e-6,
    ("energy", "cal"): 4.184,
    ("energy", "mcal"): 4.184e-3,
    # heat per mole -> J/mol
    ("heat_per_mole", "J/mol"): 1.0,
    ("heat_per_mole", "kJ/mol"): 1e3,
    ("heat_per_mole", "uJ/mol"): 1e-6,
    ("heat_per_mole", "µJ/mol"): 1e-6,
    ("heat_per_mole", "cal/mol"): 4.184,
    ("heat_per_mole", "kcal/mol"): 4184.0,
    # volume -> L
    ("volume", "L"): 1.0,
    ("volume", "mL"): 1e-3,
    ("volume", "uL"): 1e-6,
    ("volume", "µL"): 1e-6,
    # concentration -> mol/L
    ("concentration", "M"): 1.0,
    ("concentration", "mol/L"): 1.0,
    ("concentration", "mM"): 1e-3,
    ("concentration", "mmol/L"): 1e-3,
    ("concentration", "uM"): 1e-6,
    ("concentration", "µM"): 1e-6,
    ("concentration", "umol/L"): 1e-6,
    ("concentration", "nM"): 1e-9,
    ("concentration", "nmol/L"): 1e-9,
    # time -> s
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "ms"): 1e-3,
    ("time", "min"): 60.0,
    ("time", "hr"): 3600.0,
}

_CANONICAL = {
    "power": "W",
    "energy": "J",
    "heat_per_mole": "J/mol",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
    "time": "s",
}


@dataclass
class Quantity:
    raw: str
    value: float | None
    unit: str | None
    dimension: str
    canonical_value: float | None = None
    canonical_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
        }


def parse_quantity(token: str, dimension: str) -> Quantity:
    """Parse ``"310 uM"`` style tokens without dropping the raw spelling.

    Unconvertible / unparseable values return a Quantity whose canonical
    fields are ``None``; the caller attaches a positioned diagnostic and the
    raw value stays at its original location.
    """
    token = token.strip()
    match = _TOKEN_RE.match(token)
    if not match:
        return Quantity(raw=token, value=None, unit=None, dimension=dimension)
    try:
        value = float(match.group(1))
    except ValueError:
        return Quantity(raw=token, value=None, unit=None, dimension=dimension)
    unit = match.group(2)
    q = Quantity(raw=token, value=value, unit=unit, dimension=dimension)
    if math.isnan(value) or math.isinf(value):
        return q
    if dimension == "temperature":
        if unit in ("K", None):
            q.canonical_value = value
        elif unit in ("C", "degC", "°C"):
            q.canonical_value = value + 273.15
        else:
            return q
        q.canonical_unit = "K"
        return q
    if unit is None:
        return q
    factor = _FACTORS.get((dimension, unit))
    if factor is None:
        return q
    q.canonical_value = value * factor
    q.canonical_unit = _CANONICAL[dimension]
    return q


def require_canonical(
    q: Quantity, code: str, file: str | None = None, line: int | None = None
) -> float:
    if q.canonical_value is None:
        unit_part = f" unit {q.unit!r}" if q.unit else ""
        diag = Diagnostic(
            code=code,
            message=(
                f"value {q.raw!r}{unit_part} cannot be converted to canonical "
                f"{_CANONICAL[q.dimension]}; kept at its source position"
            ),
            file=file,
            line=line,
            raw=q.raw,
        )
        raise SemanticImpasse(f"unconvertible {q.dimension}", [diag])
    return q.canonical_value


def canonical_unit(dimension: str) -> str:
    return _CANONICAL[dimension]
