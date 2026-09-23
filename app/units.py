"""Unit conversion with original-representation preservation.

Every conversion returns either a converted SI value or a diagnostic; it never
silently substitutes a default.  Unconvertible values stay at their origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Unit:
    name: str
    quantity: str
    factor: float = 1.0          # multiply value -> SI
    offset: float = 0.0          # SI = value * factor + offset
    si_label: str = ""

    def to_si(self, value: float) -> float:
        return value * self.factor + self.offset


_UNITS: dict[str, Unit] = {}


def _register(unit: Unit, *aliases: str):
    for name in (unit.name, *aliases):
        _UNITS[name.strip().lower()] = unit


# time -> seconds
_register(Unit("s", "time", 1.0, si_label="s"), "sec", "secs", "second", "seconds")
_register(Unit("min", "time", 60.0, si_label="s"), "minute", "minutes")
_register(Unit("ms", "time", 1e-3, si_label="s"), "millisecond", "milliseconds")

# power -> watt
_register(Unit("W", "power", 1.0, si_label="W"), "watt", "watts")
_register(Unit("mW", "power", 1e-3, si_label="W"), "milliwatt")
_register(Unit("µW", "power", 1e-6, si_label="W"), "uW", "microwatt", "microwatts")
_register(Unit("µcal/s", "power", 4.184e-6, si_label="W"), "ucal/s", "microcal/s")
_register(Unit("cal/s", "power", 4.184, si_label="W"))
_register(Unit("mcal/s", "power", 4.184e-3, si_label="W"))

# energy -> joule
_register(Unit("J", "energy", 1.0, si_label="J"), "joule", "joules")
_register(Unit("mJ", "energy", 1e-3, si_label="J"))
_register(Unit("cal", "energy", 4.184, si_label="J"), "calorie", "calories")
_register(Unit("kcal", "energy", 4184.0, si_label="J"), "kilocalorie", "kilocalories")
_register(Unit("µcal", "energy", 4.184e-6, si_label="J"), "ucal", "microcal", "microcalorie")

# volume -> litre
_register(Unit("L", "volume", 1.0, si_label="L"), "litre", "liter")
_register(Unit("mL", "volume", 1e-3, si_label="L"), "millilitre", "milliliter")
_register(Unit("µL", "volume", 1e-6, si_label="L"), "uL", "microlitre", "microliter")

# concentration -> mol/L
_register(Unit("M", "concentration", 1.0, si_label="mol/L"), "mol/L", "molar")
_register(Unit("mM", "concentration", 1e-3, si_label="mol/L"), "mmol/L")
_register(Unit("µM", "concentration", 1e-6, si_label="mol/L"), "uM", "umol/L")
_register(Unit("nM", "concentration", 1e-9, si_label="mol/L"), "nmol/L")

# temperature -> kelvin (stored alongside Celsius original; label K)
_register(Unit("K", "temperature", 1.0, si_label="K"), "kelvin")
_register(Unit("°C", "temperature", 1.0, 273.15, si_label="K"), "C", "degC", "celsius")

# heat per amount -> J/mol
_register(Unit("J/mol", "heat_per_mol", 1.0, si_label="J/mol"))
_register(Unit("kJ/mol", "heat_per_mol", 1e3, si_label="J/mol"))
_register(Unit("cal/mol", "heat_per_mol", 4.184, si_label="J/mol"))
_register(Unit("kcal/mol", "heat_per_mol", 4184.0, si_label="J/mol"))

# affinity 1/M (identity, aliases only)
_register(Unit("M^-1", "affinity", 1.0, si_label="M^-1"), "1/M", "L/mol", "M-1")


def lookup_unit(text: Optional[str], quantity: str) -> tuple[Optional[Unit], Optional[dict]]:
    if text is None:
        return None, None
    key = text.strip().lower()
    unit = _UNITS.get(key)
    if unit is None:
        return None, {
            "code": "UNKNOWN_UNIT",
            "unit": text,
            "quantity": quantity,
            "message": f"unknown unit {text!r} for {quantity}",
        }
    if unit.quantity != quantity:
        return None, {
            "code": "UNIT_QUANTITY_MISMATCH",
            "unit": text,
            "expected_quantity": quantity,
            "actual_quantity": unit.quantity,
            "message": f"unit {text!r} is {unit.quantity}, expected {quantity}",
        }
    return unit, None


def convert(value: float, unit_text: str, quantity: str) -> tuple[Optional[float], Optional[dict]]:
    unit, diag = lookup_unit(unit_text, quantity)
    if diag is not None:
        return None, diag
    return unit.to_si(float(value)), None


def convert_field(raw_value, raw_unit: str, quantity: str, *, path: str, line=None):
    """Return (si_value_or_None, diagnostic_or_None).

    A bad numeric token or a bad unit yields a positional diagnostic and None;
    the caller keeps the original representation at the origin.
    """
    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        return None, {
            "code": "NON_NUMERIC_VALUE",
            "path": path,
            "line": line,
            "raw": raw_value,
            "quantity": quantity,
            "message": f"cannot interpret {raw_value!r} as a number for {quantity}",
        }
    si, diag = convert(number, raw_unit, quantity)
    if diag is not None:
        diag.update({"path": path, "line": line, "raw": raw_value})
        return None, diag
    return si, None
