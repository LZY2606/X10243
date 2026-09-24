"""Quantities keep their raw representation; the derived layer normalizes.

``Quantity`` stores the raw string as imported plus, when a conversion exists,
the canonical numeric value and unit. Unconvertible values stay in place with a
diagnostic instead of being replaced with zero/default.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import Diagnostic

# multiplier -> canonical unit
FACTORS: dict[tuple[str, str], tuple[float, str]] = {
    ("time", "s"): (1.0, "s"),
    ("time", "sec"): (1.0, "s"),
    ("time", "second"): (1.0, "s"),
    ("time", "min"): (60.0, "s"),
    ("time", "minute"): (60.0, "s"),
    ("power", "cal/s"): (1.0, "cal/s"),
    ("power", "ucal/s"): (1e-6, "cal/s"),
    ("power", "µcal/s"): (1e-6, "cal/s"),
    ("power", "uCal/sec"): (1e-6, "cal/s"),
    ("heat", "cal"): (1.0, "cal"),
    ("heat", "ucal"): (1e-6, "cal"),
    ("heat", "µcal"): (1e-6, "cal"),
    ("heat", "kcal"): (1e3, "cal"),
    ("heat", "j"): (1.0 / 4.184, "cal"),
    ("volume", "l"): (1.0, "L"),
    ("volume", "ml"): (1e-3, "L"),
    ("volume", "ul"): (1e-6, "L"),
    ("volume", "µl"): (1e-6, "L"),
    ("concentration", "m"): (1.0, "M"),
    ("concentration", "mm"): (1e-3, "M"),
    ("concentration", "um"): (1e-6, "M"),
    ("concentration", "µm"): (1e-6, "M"),
    ("concentration", "nm"): (1e-9, "M"),
    ("concentration", "mmolar"): (1e-3, "M"),
}

CANONICAL = {
    "time": "s",
    "power": "cal/s",
    "heat": "cal",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
}


@dataclass
class Quantity:
    raw: str
    kind: str
    value: float | None = None
    unit: str | None = None
    canonical_unit: str | None = None
    diagnostics: list | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "canonical_unit": self.canonical_unit,
            "diagnostics": [d.to_dict() for d in (self.diagnostics or [])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Quantity":
        q = cls(raw=d["raw"], kind=d["kind"], value=d.get("value"),
                unit=d.get("unit"), canonical_unit=d.get("canonical_unit"))
        q.diagnostics = [Diagnostic.from_dict(x) for x in d.get("diagnostics", [])]
        return q


def parse_quantity(raw: str, kind: str, location: str = "") -> Quantity:
    """Parse '<number> <unit>' keeping the original text.

    Temperatures need an offset (not a simple factor) and are handled
    separately. Anything unconvertible returns an ok=False Quantity carrying a
    diagnostic; callers decide whether it is fatal for their layer.
    """
    text = (raw or "").strip()
    q = Quantity(raw=text, kind=kind)
    if not text:
        q.diagnostics = [Diagnostic(
            "semantic_empty_quantity", "empty quantity value", "error",
            location, raw)]
        return q
    parts = text.split(None, 1)
    num_part = parts[0]
    unit_part = parts[1].strip() if len(parts) == 2 else ""
    try:
        num = float(num_part)
    except ValueError:
        q.diagnostics = [Diagnostic(
            "format_number", f"not a number: {num_part!r}", "error",
            f"{location}#number", raw)]
        return q
    if num != num or num in (float("inf"), float("-inf")):
        q.diagnostics = [Diagnostic(
            "semantic_non_finite", "quantity is NaN/Infinity", "error",
            f"{location}#number", raw)]
        return q

    if kind == "temperature":
        return _parse_temperature(num, unit_part, text, location)

    key = (kind, unit_part.lower())
    if key not in FACTORS:
        q.diagnostics = [Diagnostic(
            "semantic_unknown_unit",
            f"unknown {kind} unit: {unit_part!r}", "error",
            f"{location}#unit", raw)]
        return q
    factor, canon = FACTORS[key]
    q.value = num * factor
    q.unit = unit_part
    q.canonical_unit = canon
    return q


def _parse_temperature(num: float, unit: str, raw: str, location: str) -> Quantity:
    q = Quantity(raw=raw, kind="temperature", unit=unit or None)
    u = unit.strip().lower()
    if u in ("k", "kelvin"):
        q.value = num
    elif u in ("c", "°c", "degc", "celsius"):
        q.value = num + 273.15
    else:
        q.diagnostics = [Diagnostic(
            "semantic_unknown_unit", f"unknown temperature unit: {unit!r}",
            "error", f"{location}#unit", raw)]
        return q
    q.canonical_unit = "K"
    return q


def require(q: Quantity, location: str = "") -> float:
    """Hard requirement on a normalized quantity (semantic impossibility)."""
    if not q.ok:
        from .errors import SemanticImpossibility
        d = (q.diagnostics or [None])[0]
        raise SemanticImpossibility(
            f"unusable {q.kind} quantity at {location}: {q.raw!r} "
            f"({d.message if d else 'no conversion'})")
    return q.value
