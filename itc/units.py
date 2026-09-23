"""Unit registry. Raw payloads keep their original representation; conversion
only happens in the derived layer. Unconvertible values raise UnitUnconvertible
so callers can attach a located diagnostic instead of dropping the value."""

CONCENTRATION = {"mM": 1.0, "uM": 1e-3, "µM": 1e-3, "M": 1e3}
VOLUME = {"uL": 1.0, "µL": 1.0, "mL": 1e3, "L": 1e6}
POWER = {"ucal/s": 1.0, "µcal/s": 1.0, "mcal/s": 1e3, "uJ/s": 0.2388459, "µJ/s": 0.2388459}
HEAT = {"ucal": 1.0, "µcal": 1.0, "cal": 1e6, "kcal": 1e9}
TIME = {"s": 1.0, "sec": 1.0, "min": 60.0}
TEMPERATURE = {"K": ("K", 0.0), "C": ("K", 273.15), "°C": ("K", 273.15)}

FAMILIES = {
    "concentration": CONCENTRATION,
    "volume": VOLUME,
    "power": POWER,
    "heat": HEAT,
    "time": TIME,
}


class UnitUnconvertible(Exception):
    def __init__(self, unit, family):
        super().__init__(f"unit {unit!r} cannot be converted within {family}")
        self.unit = unit
        self.family = family


def convert(value, from_unit, to_unit, family):
    table = FAMILIES[family]
    if from_unit not in table:
        raise UnitUnconvertible(from_unit, family)
    if to_unit not in table:
        raise UnitUnconvertible(to_unit, family)
    return value * table[from_unit] / table[to_unit]


def temperature_to_kelvin(value, unit):
    if unit not in TEMPERATURE:
        raise UnitUnconvertible(unit, "temperature")
    _, offset = TEMPERATURE[unit]
    return value + offset
