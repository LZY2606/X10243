import math

from .errors import FormatCorrupt


UNIT_CONVERSIONS = {
    "power": {
        "uw": (1.0, "µW"),
        "µw": (1.0, "µW"),
        "μw": (1.0, "µW"),
        "mw": (1000.0, "µW"),
    },
    "time": {
        "s": (1.0, "s"),
        "sec": (1.0, "s"),
        "second": (1.0, "s"),
        "min": (60.0, "s"),
        "minute": (60.0, "s"),
    },
    "volume": {
        "ul": (1.0, "µL"),
        "µl": (1.0, "µL"),
        "μl": (1.0, "µL"),
        "ml": (1000.0, "µL"),
    },
    "concentration": {
        "um": (1.0, "µM"),
        "µm": (1.0, "µM"),
        "μm": (1.0, "µM"),
        "nm": (0.001, "µM"),
        "mm": (1000.0, "µM"),
    },
    "temperature": {
        "k": (1.0, "K"),
        "c": (None, "K"),
        "°c": (None, "K"),
    },
    "energy": {
        "ucal": (1.0, "µcal"),
        "µcal": (1.0, "µcal"),
        "μcal": (1.0, "µcal"),
        "cal": (1_000_000.0, "µcal"),
        "mj": (239.005736, "µcal"),
    },
}


def parse_quantity(raw, kind, path="value"):
    if isinstance(raw, dict):
        value = raw.get("value")
        unit = raw.get("unit")
        raw_repr = raw.get("raw", f"{value} {unit}")
    else:
        text = str(raw).strip()
        raw_repr = text
        parts = text.replace("μ", "µ").split()
        if len(parts) != 2:
            raise FormatCorrupt(f"Expected '<number> <unit>' at {path}", path)
        value, unit = parts
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FormatCorrupt(f"Non-numeric quantity at {path}", path) from exc
    if not math.isfinite(number):
        raise FormatCorrupt(f"Non-finite quantity at {path}", path)
    key = str(unit).strip().lower().replace("μ", "µ")
    table = UNIT_CONVERSIONS[kind]
    if key not in table:
        raise FormatCorrupt(f"Unsupported {kind} unit '{unit}' at {path}", path)
    factor, canonical_unit = table[key]
    if kind == "temperature" and factor is None:
        canonical = number + 273.15
    else:
        canonical = number * factor
    return {
        "raw": str(raw_repr),
        "value": number,
        "unit_input": str(unit),
        "canonical_value": canonical,
        "canonical_unit": canonical_unit,
        "path": path,
    }


def canonical(raw, kind, path="value"):
    return parse_quantity(raw, kind, path)["canonical_value"]


def display(quantity):
    if quantity is None:
        return None
    if isinstance(quantity, dict):
        return f"{quantity['canonical_value']:.6g} {quantity['canonical_unit']}"
    return str(quantity)
