"""Unit handling.

Raw materials keep their original representation. Normalization happens
only in the derived (compute) layer. Values that cannot be converted stay
in place and produce a positioned diagnostic; they are never silently
dropped or replaced by defaults.
"""

POWER_TO_UJ_S = {"uJ/s": 1.0, "uj/s": 1.0, "ucal/s": 4.184, "mcal/s": 4184.0}
TIME_TO_S = {"s": 1.0, "min": 60.0}
VOL_TO_UL = {"uL": 1.0, "mL": 1000.0}
CONC_TO_MM = {"mM": 1.0, "uM": 1e-3, "M": 1e3}


def _convert(value, unit, table, kind, path, diagnostics):
    if unit in table:
        return value * table[unit]
    diagnostics.append({
        "code": "unit_unconvertible",
        "path": path,
        "message": "cannot convert %s unit %r; raw value kept in place" % (kind, unit),
        "unit": unit,
        "raw_value": value,
    })
    return None


def power_to_uj_s(value, unit, path, diagnostics):
    return _convert(value, unit, POWER_TO_UJ_S, "power", path, diagnostics)


def time_to_s(value, unit, path, diagnostics):
    return _convert(value, unit, TIME_TO_S, "time", path, diagnostics)


def vol_to_ul(value, unit, path, diagnostics):
    return _convert(value, unit, VOL_TO_UL, "volume", path, diagnostics)


def conc_to_mm(value, unit, path, diagnostics):
    return _convert(value, unit, CONC_TO_MM, "concentration", path, diagnostics)


def temp_to_k(value, unit, path, diagnostics):
    if unit == "K":
        return float(value)
    if unit in ("C", "degC"):
        return float(value) + 273.15
    diagnostics.append({
        "code": "unit_unconvertible",
        "path": path,
        "message": "cannot convert temperature unit %r; raw value kept" % unit,
        "unit": unit,
        "raw_value": value,
    })
    return None
