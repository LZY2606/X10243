"""Unit handling.

Raw representations are preserved verbatim on ingest. Normalization to
canonical units happens only in the derived layer (pipeline). Values that
cannot be converted stay in place and carry a located diagnostic; they are
never dropped to make plotting easier.
"""
import re

from .common import diagnostic

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
    "energy": "uJ",
}

_SCALE = {
    ("time", "s"): 1.0,
    ("time", "min"): 60.0,
    ("power", "uJ/s"): 1.0,
    ("power", "ucal/s"): 4.184,
    ("volume", "uL"): 1.0,
    ("volume", "mL"): 1000.0,
    ("concentration", "uM"): 1.0,
    ("concentration", "mM"): 1000.0,
    ("energy", "uJ"): 1.0,
    ("energy", "ucal"): 4.184,
}

_ALIASES = {
    "µJ/s": "uJ/s",
    "µL": "uL",
    "µM": "uM",
    "°C": "degC",
    "deg C": "degC",
}

_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(.+?)\s*$")


def normalize_quantity(raw, dimension, location):
    """Normalize a raw quantity to the canonical unit of ``dimension``.

    Returns (value, canonical_unit, diagnostic_or_None). On failure the
    value is None and the diagnostic explains why; callers must keep the
    raw value in place and must not substitute defaults.
    """
    if raw is None:
        return None, None, diagnostic("quantity_missing", "no value supplied", location)
    if isinstance(raw, (int, float)):
        return None, None, diagnostic(
            "unit_missing", f"bare number {raw!r} carries no unit; left unconverted", location)
    text = str(raw).strip()
    m = _NUMBER.match(text)
    if not m:
        return None, None, diagnostic(
            "quantity_unparseable", f"cannot parse quantity {text!r}; left unconverted", location)
    value = float(m.group(1))
    unit = _ALIASES.get(m.group(2), m.group(2))
    if dimension == "temperature":
        if unit == "K":
            return value, "K", None
        if unit == "degC":
            return value + 273.15, "K", None
        return None, None, diagnostic(
            "unit_unknown", f"unknown temperature unit in {text!r}; left unconverted", location)
    key = (dimension, unit)
    if key not in _SCALE:
        return None, None, diagnostic(
            "unit_unknown", f"unknown {dimension} unit in {text!r}; left unconverted", location)
    return value * _SCALE[key], CANONICAL[dimension], None
