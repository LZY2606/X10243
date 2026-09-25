"""Quantity parsing with original representation preserved.

Import layer keeps the raw token on every value; only the derived layer
normalizes. A token that cannot be converted stays in place and gets a
diagnostic instead of being dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import ITCFormatError, diag

# canonical SI-ish unit used inside derived layers
TO_SI = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
             "ms": 1e-3, "min": 60.0, "m": 60.0, "h": 3600.0},
    "volume": {"l": 1.0, "ml": 1e-3, "ul": 1e-6, "µl": 1e-6,
               "liter": 1.0, "litre": 1.0},
    "concentration": {"m": 1.0, "mol/l": 1.0, "mm": 1e-3, "um": 1e-6,
                      "µm": 1e-6, "nm": 1e-9, "molar": 1.0},
    "temperature": {"c": 1.0, "°c": 1.0, "k": 1.0, "kelvin": 1.0},
    "power": {"w": 1.0, "uw": 1e-6, "µw": 1e-6, "mw": 1e-3, "cal/s": 4.184},
    "energy": {"j": 1.0, "cal": 4.184, "kcal": 4184.0, "mj": 1e-3,
               "uj": 1e-6, "µj": 1e-6},
}
UNIT_ALIASES = {"sec": "s", "second": "s", "seconds": "s",
                "liter": "l", "litre": "l", "molar": "M",
                "°c": "C", "kelvin": "K", "µl": "µL"}
_DISPLAY = {"time": "s", "volume": "L", "concentration": "M",
            "temperature": "K", "power": "W", "energy": "J"}

_TOKEN_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
                       r"\s*([^\s0-9].*?)?\s*$")


@dataclass
class Quantity:
    raw: str
    kind: str
    value: float | None = None          # canonical SI value
    unit: str | None = None             # canonical unit label
    convert_ok: bool = False
    diagnostics: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"raw": self.raw, "unit_kind": self.kind,
             "unit": _DISPLAY[self.kind], "convert_ok": self.convert_ok}
        if self.convert_ok:
            d["value"] = self.value
        if self.diagnostics:
            d["diagnostics"] = self.diagnostics
        return d


def parse_quantity(raw, kind: str, location: str) -> Quantity:
    text = "" if raw is None else str(raw)
    q = Quantity(raw=text, kind=kind)
    m = _TOKEN_RE.match(text)
    if not m:
        q.diagnostics.append(diag(
            location, "UNPARSEABLE_QUANTITY",
            f"cannot parse {kind} token", value=text))
        return q
    number = float(m.group(1))
    token_unit = (m.group(2) or "").strip()
    table = TO_SI[kind]
    if not token_unit:
        # unitless numeric token: accept only if table defines ""? It does not,
        # so require an explicit unit (domain spec: all quantities carry units).
        q.diagnostics.append(diag(
            location, "MISSING_UNIT",
            f"{kind} value must carry a unit", value=text))
        return q
    factor = table.get(token_unit.lower())
    if factor is None:
        q.diagnostics.append(diag(
            location, "UNKNOWN_UNIT",
            f"unit {token_unit!r} not convertible for {kind}",
            value=text, unit=token_unit))
        return q
    if kind == "temperature":
        if token_unit.lower() in ("k", "kelvin"):
            q.value = number
            q.unit = "K"
        else:
            q.value = number + 273.15
            q.unit = "K"
    else:
        q.value = number * factor
        q.unit = _DISPLAY[kind]
    q.convert_ok = True
    return q


def require(q: Quantity, location: str):
    """Raise FORMAT_CORRUPT with attached diagnostics when conversion failed."""
    if not q.convert_ok:
        raise ITCFormatError(f"unconvertible quantity at {location}",
                             q.diagnostics)
    return q.value


def display_unit(kind: str) -> str:
    return _DISPLAY[kind]
