"""Unit normalization for the derived layer.

Materials keep their raw representation verbatim. Normalization happens
only when deriving computations. Values that cannot be converted stay in
place in the raw material and produce a located diagnostic here; they are
never silently dropped or replaced by defaults.
"""

from .errors import diag

# kind -> (canonical unit, {unit: multiplicative scale to canonical})
SCALE_UNITS = {
    "power": ("uJ/s", {"uJ/s": 1.0, "ucal/s": 4.184, "uW": 1.0, "nW": 1e-3}),
    "volume": ("uL", {"uL": 1.0, "mL": 1000.0, "nL": 1e-3}),
    "concentration": ("mM", {"mM": 1.0, "uM": 1e-3, "M": 1000.0}),
    "time": ("s", {"s": 1.0, "min": 60.0}),
    "heat": ("uJ", {"uJ": 1.0, "ucal": 4.184, "mJ": 1000.0}),
}

# temperature is affine, handled separately
TEMPERATURE_CANONICAL = "K"


def canonical_unit(kind):
    if kind == "temperature":
        return TEMPERATURE_CANONICAL
    return SCALE_UNITS[kind][0]


def normalize_scalar(value, unit, kind, path):
    """-> (value_in_canonical | None, diagnostic | None)"""
    if value is None:
        return None, diag(path, "value_missing", f"{kind} value missing")
    if kind == "temperature":
        if unit == "K":
            return float(value), None
        if unit in ("degC", "C"):
            return float(value) + 273.15, None
        return None, diag(path, "unit_unconvertible",
                          f"temperature unit {unit!r} not convertible to K")
    if kind not in SCALE_UNITS:
        return None, diag(path, "kind_unknown", f"no unit table for kind {kind!r}")
    table = SCALE_UNITS[kind][1]
    if unit in table:
        return float(value) * table[unit], None
    return None, diag(path, "unit_unconvertible",
                      f"{kind} unit {unit!r} not convertible to {SCALE_UNITS[kind][0]}")
