from __future__ import annotations

import math
import re
from typing import Any

from .domain import malformed


TIME_S = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
}

POWER_UCAL_S = {
    "µcal/s": 1.0,
    "ucal/s": 1.0,
    "μcal/s": 1.0,
    "microcal/s": 1.0,
    "microcalorie/s": 1.0,
    "µj/s": 1.0 / 4.184,
    "uj/s": 1.0 / 4.184,
    "μj/s": 1.0 / 4.184,
    "microj/s": 1.0 / 4.184,
    "mw": 1000.0 / 4.184,
}

ENERGY_UCAL = {
    "µcal": 1.0,
    "ucal": 1.0,
    "μcal": 1.0,
    "microcal": 1.0,
    "µj": 1.0 / 4.184,
    "uj": 1.0 / 4.184,
    "μj": 1.0 / 4.184,
    "j": 1_000_000.0 / 4.184,
}

VOLUME_UL = {
    "µl": 1.0,
    "ul": 1.0,
    "μl": 1.0,
    "microl": 1.0,
    "microliter": 1.0,
    "microliters": 1.0,
    "ml": 1000.0,
    "milliliter": 1000.0,
    "milliliters": 1000.0,
    "l": 1_000_000.0,
    "liter": 1_000_000.0,
    "liters": 1_000_000.0,
}

CONCENTRATION_UM = {
    "µm": 1.0,
    "um": 1.0,
    "μm": 1.0,
    "micromolar": 1.0,
    "nm": 0.001,
    "nanomolar": 0.001,
    "mm": 1000.0,
    "millimolar": 1000.0,
    "m": 1_000_000.0,
    "molar": 1_000_000.0,
}

UNITS = {
    "time": TIME_S,
    "power": POWER_UCAL_S,
    "energy": ENERGY_UCAL,
    "volume": VOLUME_UL,
    "concentration": CONCENTRATION_UM,
}

CANONICAL_UNIT = {
    "time": "s",
    "power": "µcal/s",
    "energy": "µcal",
    "volume": "µL",
    "concentration": "µM",
    "temperature": "K",
}


def _number(token: str) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise malformed(f"Value is not numeric: {token!r}", raw_value=token) from exc
    if not math.isfinite(value):
        raise malformed("Value must be finite", raw_value=token)
    return value


def parse_quantity(raw: Any, dimension: str, path: str = "$", *, positive: bool = False) -> dict[str, Any]:
    if raw is None:
        raise malformed("Required quantity is missing", path)
    text = str(raw).strip()
    if not text:
        raise malformed("Required quantity is empty", path, raw)
    number_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s].*)?"
    match = re.fullmatch(number_pattern, text)
    if not match:
        raise malformed(f"Expected number and unit, received {text!r}", path, raw)
    value = _number(match.group(1))
    unit = (match.group(2) or "").strip()
    if dimension == "temperature":
        return _temperature(value, unit, text, path)
    table = UNITS[dimension]
    key = unit.lower().replace("μ", "µ")
    if key not in table:
        raise malformed(f"Unknown {dimension} unit {unit!r}", path, raw)
    canonical = value * table[key]
    if positive and canonical <= 0:
        raise malformed(f"{dimension.capitalize()} must be greater than zero", path, raw)
    return {
        "raw": text,
        "value": canonical,
        "unit": CANONICAL_UNIT[dimension],
        "original_unit": unit,
        "dimension": dimension,
        "path": path,
    }


def _temperature(value: float, unit: str, raw: str, path: str) -> dict[str, Any]:
    normalized = unit.lower().replace("°", "")
    if normalized in {"k", "kelvin"}:
        kelvin = value
    elif normalized in {"c", "celsius"}:
        kelvin = value + 273.15
    else:
        raise malformed(f"Unknown temperature unit {unit!r}", path, raw)
    if kelvin <= 0:
        raise malformed("Thermodynamic temperature must be greater than 0 K", path, raw)
    return {
        "raw": raw,
        "value": kelvin,
        "unit": "K",
        "original_unit": unit,
        "dimension": "temperature",
        "path": path,
    }


def quantity_or_diagnostic(raw: Any, dimension: str, path: str, *, positive: bool = False):
    try:
        return parse_quantity(raw, dimension, path, positive=positive), None
    except Exception as exc:
        payload = getattr(exc, "to_payload", lambda: {"message": str(exc)})()
        diag = payload.get("diagnostics", [{}])[0]
        diag.setdefault("code", "unconvertible_quantity")
        diag.setdefault("message", payload.get("message", str(exc)))
        diag["path"] = path
        diag["raw_value"] = raw
        return None, diag

