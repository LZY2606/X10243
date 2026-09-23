"""Domain quantities carry their original representation through import.

Only the derived layer converts into canonical SI-ish working units.  A value
that cannot be converted stays at its origin with a diagnostic instead of being
silently dropped or replaced by a default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import FormatCorrupted, Severity

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# working canonical units
CANONICAL = {
    "power": "W",
    "heat": "cal",
    "time": "s",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
}

_POWER = {"uw": 1e-6, "µw": 1e-6, "μw": 1e-6, "mw": 1e-3, "w": 1.0,
          "ucal/s": 4.184e-6, "µcal/s": 4.184e-6, "μcal/s": 4.184e-6,
          "cal/s": 4.184}
_HEAT = {"ucal": 4.184e-6, "µcal": 4.184e-6, "μcal": 4.184e-6,
         "cal": 4.184, "kcal": 4184.0, "j": 1.0, "kj": 1000.0, "mj": 1e-3}
_TIME = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "ms": 1e-3,
         "min": 60.0, "minute": 60.0, "minutes": 60.0}
_VOLUME = {"l": 1.0, "ml": 1e-3, "ul": 1e-6, "µl": 1e-6, "μl": 1e-6}
_CONC = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "μm": 1e-6,
         "nm": 1e-9, "pm": 1e-12}
_TEMP = {"k": 1.0, "c": 1.0, "celsius": 1.0, "°c": 1.0, "ºc": 1.0}

_FACTORS = {
    "power": _POWER,
    "heat": _HEAT,
    "time": _TIME,
    "volume": _VOLUME,
    "concentration": _CONC,
    "temperature": _TEMP,
}


@dataclass(frozen=True)
class Quantity:
    """A scalar plus the exact text/unit it was imported as."""

    value: float | None
    unit: str
    raw: str
    kind: str
    canonical_unit: str
    canonical_value: float | None
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "kind": self.kind,
            "canonical_unit": self.canonical_unit,
            "canonical_value": self.canonical_value,
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Quantity":
        return cls(
            value=d.get("value"),
            unit=d.get("unit", ""),
            raw=d.get("raw", ""),
            kind=d.get("kind", ""),
            canonical_unit=d.get("canonical_unit", ""),
            canonical_value=d.get("canonical_value"),
            ok=bool(d.get("ok", False)),
        )


def parse_quantity(text: Any, kind: str, *, path: str) -> Quantity:
    """Parse ``"<number> <unit>"`` preserving the original text.

    Celsius is converted to Kelvin; everything else is a linear factor.
    Unparseable values produce ``ok=False`` (never an exception, never a
    default) so the raw value remains in place for the review screen.
    """
    raw = "" if text is None else str(text).strip()
    table = _FACTORS.get(kind)
    if table is None:
        raise KeyError(f"unknown quantity kind {kind!r}")
    canon_unit = CANONICAL[kind]
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", raw)
    if not m:
        return Quantity(None, "", raw, kind, canon_unit, None, False)
    num_text, unit_text = m.group(1), m.group(2).strip()
    key = unit_text.lower()
    if not _NUM_RE.match(num_text) or key not in table:
        return Quantity(None, unit_text, raw, kind, canon_unit, None, False)
    value = float(num_text)
    if kind == "temperature":
        canonical = value + 273.15 if key in ("c", "celsius", "°c", "ºc") else value
    else:
        canonical = value * table[key]
    return Quantity(value, unit_text, raw, kind, canon_unit, canonical, True)


def parse_value(text: Any, *, path: str) -> float | None:
    """Parse a unit-less numeric sample; None raw stays missing (not zero)."""
    if text is None:
        return None
    raw = str(text).strip()
    if raw == "" or raw.lower() in {"na", "n/a", "null", "missing", "--"}:
        return None
    if not _NUM_RE.match(raw):
        raise FormatCorrupted(
            f"non-numeric sample {raw!r}", path=path, value=raw
        )
    return float(raw)


def require(q: Quantity, path: str) -> float:
    if not q.ok or q.canonical_value is None:
        raise FormatCorrupted(
            f"unconvertible {q.kind}: {q.raw!r}", path=path, value=q.raw
        )
    return q.canonical_value


def heat_in_ucal(canonical_cal: float) -> float:
    """Working helper: canonical cal -> microcalorie."""
    return canonical_cal / 4.184e-6


def ucal_to_cal(ucal: float) -> float:
    return ucal * 4.184e-6
