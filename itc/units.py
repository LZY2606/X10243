"""Quantities carry their raw representation; normalization happens only in the
derived layer.

Each parsed scalar is stored as a ``Measured`` value: the original text and unit
are preserved verbatim, and ``value``/``unit`` hold canonical SI-adjacent units
(µcal/s, µcal, µL, mM, K, s).  When conversion is impossible ``value`` stays
``None`` and a diagnostic is attached *in place* - the point is never dropped
just to keep a plot working.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any

from .errors import Diagnostic

CAL_PER_JOULE = 1.0 / 4.184

# factor -> canonical unit
_UNIT_TABLE: dict[tuple[str, str], float] = {
    # power -> µcal/s
    ("power", "µcal/s"): 1.0,
    ("power", "ucal/s"): 1.0,
    ("power", "mcal/s"): 1.0e3,
    ("power", "cal/s"): 1.0e6,
    ("power", "mw"): (1.0e-3) * CAL_PER_JOULE * 1.0e6,  # 239.0058 µcal/s
    ("power", "uw"): (1.0e-6) * CAL_PER_JOULE * 1.0e6,
    # energy -> µcal
    ("energy", "µcal"): 1.0,
    ("energy", "ucal"): 1.0,
    ("energy", "mcal"): 1.0e3,
    ("energy", "cal"): 1.0e6,
    ("energy", "mj"): 1.0e-3 * CAL_PER_JOULE * 1.0e6,
    ("energy", "uj"): 1.0e-6 * CAL_PER_JOULE * 1.0e6,
    # volume -> µL
    ("volume", "µl"): 1.0,
    ("volume", "ul"): 1.0,
    ("volume", "ml"): 1.0e3,
    ("volume", "l"): 1.0e6,
    # concentration -> mM
    ("concentration", "mm"): 1.0,
    ("concentration", "um"): 1.0e-3,
    ("concentration", "µm"): 1.0e-3,
    ("concentration", "nm"): 1.0e-6,
    ("concentration", "m"): 1.0e3,
    # time -> s
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "min"): 60.0,
    ("time", "m"): 60.0,
}

CANONICAL = {
    "power": "µcal/s",
    "energy": "µcal",
    "volume": "µL",
    "concentration": "mM",
    "time": "s",
}


@dataclass
class Measured:
    raw_text: str
    raw_unit: str | None
    kind: str
    value: float | None = None
    unit: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": {"text": self.raw_text, "unit": self.raw_unit},
            "canonical": None if self.value is None else {"value": self.value, "unit": self.unit},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def make_measured(text: str, unit: str | None, kind: str, *,
                  file: str | None = None, line: int | None = None,
                  target: str | None = None) -> Measured:
    m = Measured(raw_text=text, raw_unit=unit, kind=kind)
    try:
        num = float(text)
    except (TypeError, ValueError):
        m.diagnostics.append(Diagnostic(
            "not_a_number", f"value {text!r} is not numeric", file, line, None, target))
        return m
    if unit is None:
        m.diagnostics.append(Diagnostic(
            "unit_missing", f"{kind} value {text!r} has no unit", file, line, None, target))
        return m
    factor = _UNIT_TABLE.get((kind, unit.strip().lower()))
    if factor is None:
        m.diagnostics.append(Diagnostic(
            "unit_unconvertible",
            f"unit {unit!r} cannot be converted for kind {kind!r}",
            file, line, None, target))
        return m
    m.value = num * factor
    m.unit = CANONICAL[kind]
    return m


def parse_temperature(text: str, unit: str | None, *, file=None, line=None, target=None) -> Measured:
    m = Measured(raw_text=text, raw_unit=unit, kind="temperature")
    try:
        num = float(text)
    except (TypeError, ValueError):
        m.diagnostics.append(Diagnostic("not_a_number", f"temperature {text!r} not numeric",
                                        file, line, None, target))
        return m
    u = (unit or "").strip().lower()
    if u in ("c", "°c", "celsius"):
        m.value, m.unit = num + 273.15, "K"
    elif u in ("k", "kelvin"):
        m.value, m.unit = num, "K"
    else:
        m.diagnostics.append(Diagnostic(
            "unit_unconvertible", f"temperature unit {unit!r} not recognized",
            file, line, None, target))
    return m
