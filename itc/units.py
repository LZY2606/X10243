"""Unit handling.

Raw representations are preserved by callers (materials keep raw text);
this module only normalizes into canonical units for the derived layer.
Values that cannot be converted stay in place and produce a Diagnostic
with a position; nothing is silently dropped for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Diagnostic:
    code: str
    message: str
    position: str
    context: dict = field(default_factory=dict)

    def to_json(self):
        return asdict(self)


# raw unit -> (canonical unit, scale, offset)
POWER_UNITS = {
    "uJ/s": ("uJ/s", 1.0, 0.0), "µJ/s": ("uJ/s", 1.0, 0.0), "μJ/s": ("uJ/s", 1.0, 0.0),
    "ucal/s": ("uJ/s", 4.184, 0.0), "µcal/s": ("uJ/s", 4.184, 0.0), "μcal/s": ("uJ/s", 4.184, 0.0),
    "mJ/s": ("uJ/s", 1000.0, 0.0),
}
TIME_UNITS = {
    "s": ("s", 1.0, 0.0), "sec": ("s", 1.0, 0.0), "min": ("s", 60.0, 0.0),
}
VOLUME_UNITS = {
    "uL": ("uL", 1.0, 0.0), "µL": ("uL", 1.0, 0.0), "μL": ("uL", 1.0, 0.0),
    "mL": ("uL", 1000.0, 0.0), "nL": ("uL", 0.001, 0.0),
}
CONC_UNITS = {
    "uM": ("uM", 1.0, 0.0), "µM": ("uM", 1.0, 0.0), "μM": ("uM", 1.0, 0.0),
    "mM": ("uM", 1000.0, 0.0), "nM": ("uM", 0.001, 0.0), "M": ("uM", 1.0e6, 0.0),
}
TEMP_UNITS = {
    "degC": ("degC", 1.0, 0.0), "°C": ("degC", 1.0, 0.0), "C": ("degC", 1.0, 0.0),
    "K": ("degC", 1.0, -273.15),
}


def convert_quantity(value_raw, unit_raw, table, position):
    """Convert one scalar into canonical units.

    Returns (value, canonical_unit, diagnostic). On failure value is None
    and the diagnostic carries the position and the raw representation.
    """
    try:
        value = float(str(value_raw).strip())
    except (TypeError, ValueError):
        return None, None, Diagnostic(
            "unconvertible_value", f"cannot parse number: {value_raw!r}",
            position, {"raw": value_raw})
    unit = str(unit_raw).strip()
    if unit not in table:
        return None, None, Diagnostic(
            "unknown_unit", f"unknown unit: {unit!r}",
            position, {"raw": unit_raw, "value": value_raw})
    canon, scale, offset = table[unit]
    return value * scale + offset, canon, None
