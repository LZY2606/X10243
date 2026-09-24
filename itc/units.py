"""Quantities that preserve the original representation.

Import keeps exactly what was typed (``raw``); only the derived layer
normalizes.  A value that cannot be converted stays in place: the
:class:`Quantity` is returned together with a SEMANTIC_INVALID diagnostic
instead of being dropped or replaced with a default.
"""

from __future__ import annotations

import math
import re

from .diagnostics import SEMANTIC_INVALID

_TOKEN_RE = re.compile(
    r"^\s*(?P<value>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>[^\s()]+(?:/[^\s()]+)?)\s*$"
)

# Canonical SI-ish unit of each dimension.
_UNIT_TABLE = {
    "power": {
        "uw": ("W", 1e-6),
        "µw": ("W", 1e-6),
        "μw": ("W", 1e-6),
        "mw": ("W", 1e-5),
        "w": ("W", 1.0),
        "ucal/s": ("W", 4.184e-6),
        "μcal/s": ("W", 4.184e-6),
        "cal/s": ("W", 4.184),
    },
    "time": {
        "s": ("s", 1.0),
        "sec": ("s", 1.0),
        "min": ("s", 60.0),
        "m": ("s", 60.0),
    },
    "volume": {
        "ul": ("L", 1e-6),
        "µl": ("L", 1e-6),
        "μl": ("L", 1e-6),
        "ml": ("L", 1e-3),
        "l": ("L", 1.0),
    },
    "concentration": {
        "m": ("mol/L", 1.0),
        "mm": ("mol/L", 1e-3),
        "um": ("mol/L", 1e-9),
        "µm": ("mol/L", 1e-9),
        "μm": ("mol/L", 1e-9),
        "nm": ("mol/L", 1e-12),
        "mol/l": ("mol/L", 1.0),
        "mmol/l": ("mol/L", 1e-3),
    },
    "temperature": {
        "c": ("K", None),
        "°c": ("K", None),
        "k": ("K", None),
    },
    "energy": {
        "j": ("J", 1.0),
        "mj": ("J", 1e-3),
        "uj": ("J", 1e-6),
        "µj": ("J", 1e-6),
        "μj": ("J", 1e-6),
        "cal": ("J", 4.184),
        "kcal": ("J", 4184.0),
    },
    "energy_per_mol": {
        "j/mol": ("J/mol", 1.0),
        "kj/mol": ("J/mol", 1e3),
        "cal/mol": ("J/mol", 4.184),
        "kcal/mol": ("J/mol", 4184.0),
    },
}

_DIMENSION_OF_UNIT = {}
for _dim, _units in _UNIT_TABLE.items():
    for _u in _units:
        _DIMENSION_OF_UNIT[_u] = _dim


class Quantity:
    """A typed scalar that remembers how it was written.

    ``value`` is normalized (SI) and may be ``None`` when conversion
    failed; ``raw`` and ``unit`` are always populated.
    """

    __slots__ = ("raw", "number", "unit", "dimension", "canonical_unit", "value")

    def __init__(self, raw, number, unit, dimension, canonical_unit, value):
        self.raw = raw
        self.number = number
        self.unit = unit
        self.dimension = dimension
        self.canonical_unit = canonical_unit
        self.value = value

    def to_dict(self):
        return {
            "raw": self.raw,
            "number": self.number,
            "unit": self.unit,
            "dimension": self.dimension,
            "canonical_unit": self.canonical_unit,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["raw"], d["number"], d["unit"], d["dimension"],
            d["canonical_unit"], d["value"],
        )

    def __repr__(self):
        return f"Quantity({self.raw!r} -> {self.value} {self.canonical_unit})"


def parse_quantity(text, expected_dimension=None, position=None):
    """Parse ``"0.30 mM"``-style tokens, preserving the original.

    Returns ``(Quantity | None, diagnostic | None)``.  When the token is
    not a number+unit at all, ``Quantity`` is ``None``; when the unit is
    unknown or the dimension mismatches, the quantity is returned with
    ``value=None`` plus a SEMANTIC_INVALID diagnostic so the raw value is
    never lost.
    """

    from .diagnostics import diagnostic

    if text is None:
        return None, diagnostic(
            SEMANTIC_INVALID, "missing value", at=position
        )
    raw = str(text).strip()
    match = _TOKEN_RE.match(raw)
    if not match:
        return None, diagnostic(
            SEMANTIC_INVALID,
            f"not a number+unit token: {raw!r}",
            at=position,
        )
    number = float(match.group("value"))
    unit = match.group("unit").lower()
    dimension = _DIMENSION_OF_UNIT.get(unit)
    if dimension is None:
        q = Quantity(raw, number, match.group("unit"), None, None, None)
        return q, diagnostic(
            SEMANTIC_INVALID, f"unknown unit {match.group('unit')!r}", at=position
        )
    if expected_dimension is not None and dimension != expected_dimension:
        q = Quantity(raw, number, match.group("unit"), dimension, None, None)
        return q, diagnostic(
            SEMANTIC_INVALID,
            f"expected {expected_dimension}, got {dimension}",
            at=position,
        )
    canonical_unit, factor = _UNIT_TABLE[dimension][unit]
    if dimension == "temperature":
        value = number + 273.15 if unit in ("c", "°c") else number
    else:
        value = number * factor
    return Quantity(raw, number, match.group("unit"), dimension, canonical_unit, value), None


def convert_temperature(q):
    if q is None or q.value is None:
        return None
    return q.value


def is_finite(q):
    return q is not None and q.value is not None and math.isfinite(q.value)
