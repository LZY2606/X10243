"""Unit / coordinate-basis handling.

Raw materials keep their original representation.  Normalisation happens
only in the derived layer, through `convert`.  A value that cannot be
converted stays in place and produces a located diagnostic; nothing is
silently dropped or replaced by a default.
"""

from __future__ import annotations

# linear conversions: value_to = value_from * factor
LINEAR = {
    ("uJ/s", "uJ/s"): 1.0,
    ("ucal/s", "uJ/s"): 4.184,
    ("mJ/s", "uJ/s"): 1000.0,
    ("s", "s"): 1.0,
    ("min", "s"): 60.0,
    ("uL", "uL"): 1.0,
    ("mL", "uL"): 1000.0,
    ("mL", "mL"): 1.0,
    ("uL", "mL"): 1e-3,
    ("mM", "mM"): 1.0,
    ("uM", "mM"): 1e-3,
    ("M", "mM"): 1000.0,
    ("K", "K"): 1.0,
}

# affine conversions: value_to = a * value_from + b
AFFINE = {
    ("C", "K"): (1.0, 273.15),
    ("F", "K"): (5.0 / 9.0, 255.3722222222222),
}


def convert(value, from_unit: str, to_unit: str, location: str = ""):
    """Return (converted_value, diagnostic).  Exactly one of the two is
    None.  The diagnostic carries the location of the offending value."""
    key = (from_unit, to_unit)
    if key in LINEAR:
        return value * LINEAR[key], None
    if key in AFFINE:
        a, b = AFFINE[key]
        return a * value + b, None
    return None, {
        "code": "unit_unconvertible",
        "location": location,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "value": value,
        "message": f"cannot convert {value!r} from '{from_unit}' to "
                   f"'{to_unit}'; value left in raw position",
    }


def convert_series(values, from_unit: str, to_unit: str, location: str = ""):
    """Convert a list; the first unconvertible value aborts with a
    located diagnostic (the raw series stays untouched upstream)."""
    out = []
    for i, v in enumerate(values):
        cv, diag = convert(v, from_unit, to_unit, f"{location}[{i}]")
        if diag:
            return None, diag
        out.append(cv)
    return out, None
