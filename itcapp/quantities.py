"""Quantities with units.

Rule: raw text is preserved verbatim at import (raw_repr). Normalisation to
canonical SI units happens only in the derived layer. Values that cannot be
converted stay in place and produce diagnostics; they are never silently
replaced by defaults.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# canonical unit per dimension
CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "energy": "uJ",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "count": "1",
}

# unit -> (dimension, factor to canonical)
UNITS = {
    "s": ("time", 1.0),
    "sec": ("time", 1.0),
    "min": ("time", 60.0),
    "uJ/s": ("power", 1.0),
    "uW": ("power", 1.0),
    "ucal/s": ("power", 4.184),
    "uJ": ("energy", 1.0),
    "ucal": ("energy", 4.184),
    "uL": ("volume", 1.0),
    "mL": ("volume", 1000.0),
    "mM": ("concentration", 1.0),
    "uM": ("concentration", 1e-3),
    "M": ("concentration", 1e3),
    "K": ("temperature", 1.0),
    "degC": ("temperature", None),  # affine, handled specially
    "1": ("count", 1.0),
}

TOKEN_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z0-9_/*^.-]*)\s*$")
HEADER_RE = re.compile(r"^(?P<name>[^()\[\]]+?)\s*(?:[\(\[](?P<unit>[^)\]]+)[)\]])?\s*$")


@dataclass
class Quantity:
    raw: str
    value: Optional[float]
    unit: Optional[str]
    dimension: Optional[str]
    ok: bool
    error: Optional[str] = None
    pos: Optional[str] = None  # e.g. "POWER:12"

    def to_canonical(self) -> Optional[float]:
        """Return value in canonical unit for its dimension, or None."""
        if not self.ok or self.value is None or self.unit is None:
            return None
        entry = UNITS.get(self.unit)
        if entry is None:
            return None
        _, factor = entry
        if self.unit == "degC":
            return self.value + 273.15
        if factor is None:
            return None
        return self.value * factor


def parse_quantity(token: str, default_unit: Optional[str] = None, pos: Optional[str] = None) -> Quantity:
    """Parse a raw token like '2.5 uL' or '298.15'. Never raises."""
    raw = token
    token = (token or "").strip()
    if token == "" or token.upper() in {"NA", "N/A", "NAN", "NONE", "-"}:
        return Quantity(raw=raw, value=None, unit=None, dimension=None, ok=False,
                        error="empty_or_sentinel", pos=pos)
    m = TOKEN_RE.match(token)
    if not m:
        return Quantity(raw=raw, value=None, unit=None, dimension=None, ok=False,
                        error="unparseable_number", pos=pos)
    value = float(m.group(1))
    unit = m.group(2) or default_unit
    if unit is None or unit == "":
        return Quantity(raw=raw, value=value, unit=None, dimension=None, ok=False,
                        error="missing_unit", pos=pos)
    if unit not in UNITS:
        return Quantity(raw=raw, value=value, unit=unit, dimension=None, ok=False,
                        error=f"unknown_unit:{unit}", pos=pos)
    dim = UNITS[unit][0]
    return Quantity(raw=raw, value=value, unit=unit, dimension=dim, ok=True, pos=pos)


def parse_header(cell: str):
    """'power (uJ/s)' -> ('power', 'uJ/s'); 'time_s' -> ('time', 's');
    'power_uJ_per_s' -> ('power', 'uJ/s')."""
    m = HEADER_RE.match(cell.strip())
    if not m:
        return cell.strip(), None
    name, unit = m.group("name").strip(), m.group("unit")
    if unit is None and "_" in name:
        base, _, suffix = name.partition("_")
        candidate = suffix.replace("_per_", "/")
        if candidate in UNITS:
            return base, candidate
    return name, unit


@dataclass
class QuantityColumn:
    """A column of quantities with a declared dimension."""
    name: str
    dimension: str
    unit: str  # declared unit (canonical for dimension preferred)
    items: list = field(default_factory=list)  # list[Quantity]

    def canonical_values(self):
        """list of Optional[float] in canonical units."""
        return [q.to_canonical() for q in self.items]

    def mask(self):
        """True where usable."""
        return [q.ok and q.to_canonical() is not None for q in self.items]
