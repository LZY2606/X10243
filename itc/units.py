"""Unit registry.

Raw representations are preserved at import; normalization happens only in the
derived layer.  Values that cannot be converted stay in place and produce a
diagnostic carrying the position of the offending value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# canonical unit per quantity kind
CANONICAL = {
    "power": "uJ/s",
    "time": "s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "cell_volume": "mL",
    "heat": "kJ/mol",
}

# multiplier to convert a unit into the canonical unit of its kind
_TABLE = {
    "power": {"uJ/s": 1.0, "mcal/s": 4184.0, "uW": 1.0, "mW": 1000.0},
    "time": {"s": 1.0, "min": 60.0},
    "volume": {"uL": 1.0, "nL": 0.001, "mL": 1000.0},
    "concentration": {"mM": 1.0, "uM": 0.001, "M": 1000.0},
    "temperature": {"K": 1.0},
    "cell_volume": {"mL": 1.0, "uL": 0.001},
    "heat": {"kJ/mol": 1.0, "kcal/mol": 4.184},
}


@dataclass
class Diagnostic:
    code: str
    message: str
    position: str  # machine-locatable position, e.g. "samples[17]" or "injections[3].volume"

    def as_dict(self):
        return {"code": self.code, "message": self.message, "position": self.position}


@dataclass
class NormResult:
    ok: bool
    values: list = field(default_factory=list)   # canonical values (None where unconvertible)
    raw: list = field(default_factory=list)      # original (value, unit) pairs, always kept
    diagnostics: list = field(default_factory=list)


def convert(value: float, unit: str, kind: str):
    """Return canonical value or None when the unit is unknown for the kind."""
    table = _TABLE.get(kind, {})
    if unit not in table:
        return None
    return value * table[unit]


def normalize_series(pairs, kind: str, base_position: str) -> NormResult:
    """pairs: list of [value, unit].  Never drops unconvertible values."""
    res = NormResult(ok=True)
    for i, pair in enumerate(pairs):
        value, unit = pair[0], pair[1]
        res.raw.append({"value": value, "unit": unit})
        pos = f"{base_position}[{i}]"
        if value is None:
            res.ok = False
            res.values.append(None)
            res.diagnostics.append(Diagnostic(
                "value_missing", f"missing value for {kind}", pos))
            continue
        conv = convert(value, unit, kind)
        if conv is None:
            res.ok = False
            res.values.append(None)
            res.diagnostics.append(Diagnostic(
                "unit_unconvertible",
                f"unit {unit!r} cannot be converted to {CANONICAL.get(kind, '?')} "
                f"for kind {kind!r}; raw value left in place", pos))
        else:
            res.values.append(conv)
    return res
