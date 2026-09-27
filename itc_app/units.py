"""Domain quantities: every value carries a unit (or coordinate/time basis).

Design rule (per spec): the import layer keeps the *original textual*
representation; only the derived layer normalizes into canonical SI-ish units.
Values that cannot be converted stay in place and get a diagnostic; they are
never dropped to make a plot work.
"""
from __future__ import annotations

from dataclasses import dataclass

# canonical unit per physical kind
_UNIT_FACTORS: dict[tuple[str, str], float] = {
    # (kind, unit) -> multiplier to canonical unit
    ("power", "uW"): 1.0,
    ("power", "µW"): 1.0,
    ("power", "mcal/s"): 4.184,
    ("power", "cal/s"): 4184.0,
    ("energy", "uJ"): 1.0,
    ("energy", "µJ"): 1.0,
    ("energy", "cal"): 4.184,
    ("energy", "mcal"): 4.184e-3,
    ("energy", "kcal"): 4184.0,
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "min"): 60.0,
    ("volume", "L"): 1.0,
    ("volume", "mL"): 1e-3,
    ("volume", "ml"): 1e-3,
    ("volume", "uL"): 1e-6,
    ("volume", "µL"): 1e-6,
    ("volume", "ul"): 1e-6,
    ("concentration", "M"): 1.0,
    ("concentration", "mM"): 1e-3,
    ("concentration", "uM"): 1e-6,
    ("concentration", "µM"): 1e-6,
    ("concentration", "nM"): 1e-9,
    ("temperature", "K"): 1.0,
    ("temperature", "C"): 1.0,   # offset handled separately
    ("temperature", "celsius"): 1.0,
    ("molar_energy", "cal/mol"): 1.0,
    ("molar_energy", "kcal/mol"): 1e3,
    ("molar_energy", "J/mol"): 1.0 / 4.184,
    ("molar_energy", "kJ/mol"): 1000.0 / 4.184,
}
CANONICAL = {
    "power": "uW",
    "energy": "uJ",
    "time": "s",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
    "molar_energy": "cal/mol",
}


@dataclass(frozen=True)
class Quantity:
    """A measured value that remembers how it was written."""

    value: float
    unit: str
    kind: str
    raw: str | None = None

    def canonical_value(self) -> float:
        if self.kind == "temperature":
            if self.unit.upper() in ("K",):
                return self.value
            return self.value + 273.15
        return self.value * _UNIT_FACTORS[(self.kind, self.unit)]

    def to(self, unit: str) -> float:
        if self.kind == "temperature":
            k = self.canonical_value()
            if unit.upper() == "K":
                return k
            return k - 273.15
        return self.canonical_value() / _UNIT_FACTORS[(self.kind, unit)]

    def as_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "kind": self.kind,
                "raw": self.raw, "canonical_unit": CANONICAL[self.kind],
                "canonical_value": self.canonical_value()}


def convertable(kind: str, unit: str) -> bool:
    if kind == "temperature":
        return unit.upper() in ("K", "C", "CELSIUS")
    return (kind, unit) in _UNIT_FACTORS


def parse_quantity(text: str, kind: str, default_unit: str | None = None) -> Quantity:
    """Parse '12.3 uL'. Raises ValueError for unparseable numbers/units.

    A missing unit is allowed only when ``default_unit`` is supplied (the
    column header unit). Unknown units raise ValueError so the caller can keep
    the raw text and attach a diagnostic instead of guessing.
    """
    raw = text
    s = text.strip()
    if not s:
        raise ValueError("empty value")
    # split number from unit: number prefix then unit suffix
    i = 0
    if s[0] in "+-":
        i = 1
    seen_e = False
    while i < len(s):
        ch = s[i]
        if ch.isdigit() or ch == ".":
            i += 1
        elif ch in "eE" and not seen_e:
            seen_e = True
            i += 1
            if i < len(s) and s[i] in "+-":
                i += 1
        else:
            break
    num_part, unit_part = s[:i], s[i:].strip()
    if not num_part:
        raise ValueError(f"not a number: {text!r}")
    try:
        value = float(num_part)
    except ValueError as exc:
        raise ValueError(f"not a number: {text!r}") from exc
    unit = unit_part or (default_unit or "")
    if not unit:
        raise ValueError(f"missing unit: {text!r}")
    if not convertable(kind, unit):
        raise ValueError(f"unknown {kind} unit: {unit!r}")
    return Quantity(value, unit, kind, raw=raw)
