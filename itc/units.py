"""Unit and coordinate handling.

Domain rule: original textual representations are preserved on import.
Normalisation to SI happens only on a derived (`canonical`) field; a value that
cannot be converted stays in place with a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

# Multiplicative conversion to SI for accepted units.
_UNIT_TABLE: dict[tuple[str, str], float] = {
    ("power", "uW"): 1e-6,
    ("power", "uw"): 1e-6,
    ("power", "cal/s"): 4.184,
    # time
    ("time", "s"): 1.0,
    ("time", "sec"): 1.0,
    ("time", "min"): 60.0,
    # energy
    ("energy", "uJ"): 1e-6,
    ("energy", "uj"): 1e-6,
    ("energy", "ucal"): 4.184e-6,
    ("energy", "mcal"): 4.184e-3,
    ("energy", "cal"): 4.184,
    ("energy", "J"): 1.0,
    # molar energy
    ("molar_energy", "kcal/mol"): 4184.0,
    ("energy_per_mole", "kcal/mol"): 4184.0,
    ("molar_energy", "cal/mol"): 4.184,
    ("energy_per_mole", "cal/mol"): 4.184,
    ("molar_energy", "kJ/mol"): 1000.0,
    ("energy_per_mole", "kJ/mol"): 1000.0,
    # volume
    ("volume", "uL"): 1e-9,  # m^3
    ("volume", "L"): 1e-3,
    ("volume", "mL"): 1e-6,
    # concentration
    ("concentration", "uM"): 1e-6,  # mol / m^3 == 1e-6 mol/L
    ("concentration", "mM"): 1e-3,
    ("concentration", "M"): 1.0,
    # temperature offsets
    ("temperature", "C"): 1.0,
    ("temperature", "K"): 1.0,
}


@dataclass
class Quantity:
    raw: str
    kind: str
    unit: str | None
    value: float | None
    canonical_si: float | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "unit": self.unit,
            "value": self.value,
            "canonical_si": self.canonical_si,
            "diagnostics": self.diagnostics,
        }


def parse_quantity(raw: Any, kind: str, *, location: str) -> Quantity:
    """Parse `<number><unit>` while keeping the original text."""
    text = "" if raw is None else str(raw).strip()
    q = Quantity(raw=text, kind=kind, unit=None, value=None)
    if text == "":
        q.diagnostics.append(
            {"code": "missing_value", "kind": kind, "location": location,
             "message": "Value is empty; no default was substituted."}
        )
        return q
    # split number from trailing unit: e.g. "12.3uM", "-4.2 kcal/mol", "25 C"
    idx = len(text)
    for i, ch in enumerate(text):
        if ch.isalpha():
            idx = i
            break
    num_part = text[:idx].strip()
    unit_part = text[idx:].strip()
    q.unit = unit_part or None
    try:
        q.value = float(num_part)
    except ValueError:
        q.diagnostics.append(
            {"code": "unparseable_number", "kind": kind, "raw": text,
             "location": location, "message": f"Could not parse number from {text!r}."}
        )
        return q
    if math.isnan(q.value) or math.isinf(q.value):
        q.diagnostics.append(
            {"code": "non_finite_number", "kind": kind, "raw": text,
             "location": location, "message": "Value must be finite."}
        )
        q.value = None
        return q
    if unit_part not in _ACCEPTED_UNITS.get(kind, set()):
        q.diagnostics.append(
            {"code": "unknown_unit", "kind": kind, "unit": unit_part,
             "location": location,
             "message": f"Unit {unit_part!r} is not a known {kind} unit; value kept in place."}
        )
        return q
    factor = _UNIT_TABLE[(kind, unit_part)]
    if kind == "temperature":
        # canonical is Kelvin for Celsius inputs, Kelvin as-is.
        q.canonical_si = q.value + 273.15 if unit_part in ("C", "c", "°C") else q.value
    else:
        q.canonical_si = q.value * factor
    return q


_ACCEPTED_UNITS: dict[str, set[str]] = {}
for (_kind, _unit) in _UNIT_TABLE:
    _ACCEPTED_UNITS.setdefault(_kind, set()).add(_unit)
_ACCEPTED_UNITS["temperature"].update({"c", "°C"})
# Keep _UNIT_TABLE keys lookup tolerant for Celsius variants.
for _variant in ("c", "°C"):
    _UNIT_TABLE[("temperature", _variant)] = 1.0


def to_si(raw: Any, kind: str, *, location: str) -> tuple[float | None, Quantity]:
    q = parse_quantity(raw, kind, location=location)
    return q.canonical_si, q


def convert_from_si(si_value: float, kind: str, unit: str) -> float:
    if kind == "temperature":
        return si_value - 273.15 if unit in ("C", "c", "°C") else si_value
    factor = _UNIT_TABLE.get((kind, unit))
    if factor is None:
        raise ValueError(f"Unknown unit {unit!r} for {kind}")
    return si_value / factor


def heat_uJ_from_si(area_si: float) -> float:
    """Energy (J) -> micro-joules, the natural ITC reporting unit."""
    return area_si / 1e-6
