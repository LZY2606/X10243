"""Quantities with units / coordinate references.

Raw representations are preserved on import; normalization happens only in the
derived layer. Values that cannot be converted stay in place with a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

# Conversion factors are intentionally explicit tables rather than a third party
# dependency, so the laboratory conventions used by the case pack are auditable.
_TO_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
               "min": 60.0, "minute": 60.0, "minutes": 60.0}
_TO_LITRES = {"l": 1.0, "litre": 1.0, "liters": 1.0, "liter": 1.0,
              "ml": 1e-3, "millilitre": 1e-3, "milliliter": 1e-3,
              "ul": 1e-6, "µl": 1e-6, "uL": 1e-6,
              "microlitre": 1e-6, "microliter": 1e-6}
_TO_KELVIN_OFFSET = {"c": 273.15, "°c": 273.15, "celsius": 273.15,
                     "k": 0.0, "kelvin": 0.0}
_TO_MOLAR = {"m": 1.0, "mol/l": 1.0, "mol per l": 1.0, "molar": 1.0,
             "mm": 1e-3, "mmol/l": 1e-3, "millimolar": 1e-3,
             "um": 1e-6, "µm": 1e-6, "micromolar": 1e-6,
             "nm": 1e-9, "nanomolar": 1e-9}
# ITC instruments commonly report µcal/s; 1 cal = 4.184 J exactly.
_TO_WATT = {"w": 1.0, "watt": 1.0, "watts": 1.0, "mw": 1e-3, "milliwatt": 1e-3,
            "ucal/s": 4.184e-6, "µcal/s": 4.184e-6, "microcal/s": 4.184e-6,
            "cal/s": 4.184, "j/s": 1.0}
_TO_JOULE = {"j": 1.0, "joule": 1.0, "joules": 1.0,
             "cal": 4.184, "calorie": 4.184,
             "ucal": 4.184e-6, "µcal": 4.184e-6, "microcalorie": 4.184e-6,
             "mj": 1e-3, "millijoule": 1e-3, "kj": 1e3, "kilojoule": 1e3}

_TABLES = {
    "time": (_TO_SECONDS, "s"),
    "volume": (_TO_LITRES, "L"),
    "temperature": (_TO_KELVIN_OFFSET, "K"),
    "concentration": (_TO_MOLAR, "M"),
    "power": (_TO_WATT, "W"),
    "energy": (_TO_JOULE, "J"),
}


@dataclass(frozen=True)
class Quantity:
    """A scalar carrying its original textual representation and unit token."""

    raw: str
    unit: str
    value: float | None
    kind: str
    diagnostic: str | None = None

    def normalized(self) -> float:
        if self.value is None:
            raise ValueError(self.diagnostic or f"unconvertible value: {self.raw!r} {self.unit!r}")
        table, _ = _TABLES[self.kind]
        return self.value * table[self.unit.lower()]

    def to_display(self) -> str:
        if self.value is None:
            return f"{self.raw} {self.unit}"
        _, canonical = _TABLES[self.kind]
        if self.kind == "temperature":
            table, _ = _TABLES[self.kind]
            shown = self.value + table[self.unit.lower()]
            return f"{shown:.4g} {canonical}"
        return f"{self.normalized():.6g} {canonical}"


def parse_quantity(raw, unit, kind: str) -> Quantity:
    """Parse without dropping the original representation.

    Unconvertible values are kept at their original location with a diagnostic;
    they never silently become zero / None / a default.
    """
    unit = (unit or "").strip()
    table = _TABLES.get(kind, (None, None))[0]
    if table is None:
        return Quantity(str(raw), unit, None, kind, f"unknown quantity kind {kind!r}")
    if unit.lower() not in table:
        return Quantity(str(raw), unit, None, kind,
                        f"unrecognized {kind} unit {unit!r}; value kept raw")
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return Quantity(str(raw), unit, None, kind,
                        f"non-numeric {kind} value {str(raw)!r}; value kept raw")
    if value != value or value in (float("inf"), float("-inf")):
        return Quantity(str(raw), unit, None, kind,
                        f"non-finite {kind} value {str(raw)!r}; value kept raw")
    return Quantity(str(raw), unit, value, kind)
