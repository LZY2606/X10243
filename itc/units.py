"""Unit handling.

Raw imported representation is always preserved by the callers (store layer keeps
``raw_json`` verbatim).  This module only derives canonical values for the
derived/computation layer.  Values that cannot be converted stay in place (as
``None`` in the normalized layer) and carry a located diagnostic; nothing is
silently dropped or defaulted.
"""

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "volume": "uL",
    "cell_volume": "mL",
    "concentration": "mM",
    "energy": "uJ",
    "temperature": "K",
}

_TABLES = {
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0},
    "power": {"uJ/s": 1.0, "µJ/s": 1.0, "μJ/s": 1.0, "mJ/s": 1e3, "J/s": 1e6},
    "volume": {"uL": 1.0, "µL": 1.0, "μL": 1.0, "mL": 1e3, "L": 1e6},
    "cell_volume": {"mL": 1.0, "uL": 1e-3, "µL": 1e-3, "μL": 1e-3, "L": 1e3},
    "concentration": {"mM": 1.0, "M": 1e3, "uM": 1e-3, "µM": 1e-3, "μM": 1e-3},
    "energy": {"uJ": 1.0, "µJ": 1.0, "μJ": 1.0, "mJ": 1e3, "J": 1e6},
}


def diagnostic(path, code, message, raw=None):
    d = {"path": path, "code": code, "message": message}
    if raw is not None:
        d["raw"] = raw
    return d


def convert_scalar(value, unit, quantity, path):
    """Return (canonical_value_or_None, diagnostic_or_None)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, diagnostic(path, "value_not_numeric",
                                f"value {value!r} is not numeric", raw=value)
    if quantity == "temperature":
        if unit == "K":
            return float(value), None
        if unit in ("C", "°C", "degC"):
            return float(value) + 273.15, None
        return None, diagnostic(path, "unit_unknown",
                                f"unknown temperature unit {unit!r}", raw=value)
    table = _TABLES[quantity]
    if unit in table:
        return float(value) * table[unit], None
    return None, diagnostic(path, "unit_unknown",
                            f"unknown {quantity} unit {unit!r}", raw=value)


def convert_series(values, unit, quantity, path):
    """Convert a list; unconvertible entries stay as None with located diagnostics."""
    out, diags = [], []
    for i, v in enumerate(values):
        cv, d = convert_scalar(v, unit, quantity, f"{path}[{i}]")
        out.append(cv)
        if d:
            diags.append(d)
    return out, diags
