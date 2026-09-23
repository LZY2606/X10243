"""Quantities keep their raw representation; only the derived layer normalises.

Every converted value is a :class:`QValue` carrying:

* ``raw``   - exactly what the source file said (never dropped),
* ``value`` - numeric value in the canonical unit of its dimension, or None,
* ``unit``  - canonical unit token,
* ``source_unit`` - the unit token as declared,
* ``basis`` - coordinate/reference basis when relevant (e.g. ``time:experiment_t0``),
* ``diagnostics`` - why a value could not be converted (it then stays in place).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .errors import Diagnostic

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# dimension -> canonical unit
CANONICAL = {
    "power": "uW",
    "volume": "uL",
    "time": "s",
    "concentration": "mM",
    "temperature": "K",
    "heat": "ucal",
    "molar_energy": "cal/mol",
    "energy_rate": "ucal/s",
}

# (dimension, source unit token) -> (factor to canonical, offset added after factor)
_UNITS: dict[tuple[str, str], tuple[float, float]] = {
    ("power", "uw"): (1.0, 0.0),
    ("power", "mw"): (1000.0, 0.0),
    ("power", "ucal/s"): (4.184, 0.0),  # 1 cal/s = 4.184 W -> 1 ucal/s = 4.184 uW
    ("volume", "ul"): (1.0, 0.0),
    ("volume", "ml"): (1000.0, 0.0),
    ("volume", "l"): (1e6, 0.0),
    ("time", "s"): (1.0, 0.0),
    ("time", "sec"): (1.0, 0.0),
    ("time", "min"): (60.0, 0.0),
    ("concentration", "mm"): (1.0, 0.0),
    ("concentration", "um"): (0.001, 0.0),
    ("concentration", "m"): (1000.0, 0.0),
    ("temperature", "k"): (1.0, 0.0),
    ("temperature", "c"): (1.0, 273.15),
    ("temperature", "degc"): (1.0, 273.15),
    ("heat", "ucal"): (1.0, 0.0),
    ("heat", "cal"): (1e6, 0.0),
    ("heat", "j"): (1.0 / 4.184, 0.0),  # J -> cal -> ucal
    ("molar_energy", "cal/mol"): (1.0, 0.0),
    ("molar_energy", "kcal/mol"): (1000.0, 0.0),
    ("molar_energy", "j/mol"): (1.0 / 4.184, 0.0),
    ("molar_energy", "kj/mol"): (1000.0 / 4.184, 0.0),
}


@dataclass
class QValue:
    raw: str
    dimension: str
    value: float | None = None
    unit: str | None = None
    source_unit: str | None = None
    basis: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None and not any(d.severity == "error" for d in self.diagnostics)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "dimension": self.dimension,
            "value": self.value,
            "unit": self.unit,
            "source_unit": self.source_unit,
            "basis": self.basis,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QValue":
        return cls(
            raw=data["raw"],
            dimension=data["dimension"],
            value=data.get("value"),
            unit=data.get("unit"),
            source_unit=data.get("source_unit"),
            basis=data.get("basis"),
            diagnostics=[Diagnostic.from_dict(d) for d in data.get("diagnostics", [])],
        )


def _split_number(text: str) -> tuple[str | None, str]:
    text = text.strip()
    if not text:
        return None, text
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[\s]*(.*)$", text)
    if not m:
        return None, text
    return m.group(1), m.group(2).strip()


def parse_quantity(text: str, dimension: str, location: str = "?", basis: str | None = None) -> QValue:
    """Parse ``"<number> <unit>"`` preserving raw text and attaching diagnostics."""
    raw_text = text
    qv = QValue(raw=raw_text, dimension=dimension, basis=basis)
    num_part, unit_part = _split_number(text)
    if num_part is None:
        qv.diagnostics.append(
            Diagnostic("unconvertible_value", f"not a number: {text!r}", location, "error", raw_text)
        )
        return qv
    if not _NUM_RE.match(num_part):
        qv.diagnostics.append(
            Diagnostic("unconvertible_value", f"not a number: {num_part!r}", location, "error", raw_text)
        )
        return qv
    try:
        num = float(num_part)
        if math.isnan(num) or math.isinf(num):
            raise ValueError
    except ValueError:
        qv.diagnostics.append(
            Diagnostic("unconvertible_value", f"non-finite number: {num_part!r}", location, "error", raw_text)
        )
        return qv

    unit_token = unit_part.replace("µ", "u").replace("μ", "u")
    unit_norm = unit_token.lower().replace(" ", "").replace("/", "/")
    # normalise common symbols
    unit_norm = unit_norm.replace("μ", "u")
    factor_offset = _UNITS.get((dimension, unit_norm))
    if factor_offset is None:
        qv.value = num
        qv.source_unit = unit_part or None
        qv.diagnostics.append(
            Diagnostic(
                "unknown_unit",
                f"unit {unit_part!r} is not known for {dimension}; raw value retained, not normalised",
                location,
                "error",
                raw_text,
            )
        )
        return qv

    factor, offset = factor_offset
    qv.value = num * factor + offset
    qv.unit = CANONICAL[dimension]
    qv.source_unit = unit_part or None
    return qv


def plain(text: str, dimension: str, location: str = "?", basis: str | None = None) -> QValue:
    """Parse; raise SemanticImpossible if conversion failed (convenience for required fields)."""
    from .errors import SemanticImpossible

    qv = parse_quantity(text, dimension, location, basis)
    if not qv.ok:
        d = qv.diagnostics[0]
        raise SemanticImpossible(d.message, d.location, raw_text)
    return qv


def qraw(value: float | None, dimension: str, raw: str | None = None,
         source_unit: str | None = None, basis: str | None = None) -> QValue:
    return QValue(
        raw=raw if raw is not None else (f"{value}" if value is not None else ""),
        dimension=dimension,
        value=value,
        unit=CANONICAL[dimension] if value is not None else None,
        source_unit=source_unit or CANONICAL[dimension],
        basis=basis,
    )
