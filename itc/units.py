from __future__ import annotations

import math
from typing import Any

from .errors import SemanticInvalid


UNIT_ALIASES = {
    "": (1.0, "dimensionless"),
    "1": (1.0, "dimensionless"),
    "s": (1.0, "s"),
    "sec": (1.0, "s"),
    "seconds": (1.0, "s"),
    "µL": (1.0, "µL"),
    "uL": (1.0, "µL"),
    "ul": (1.0, "µL"),
    "microliter": (1.0, "µL"),
    "µJ/s": (1.0, "µJ/s"),
    "uJ/s": (1.0, "µJ/s"),
    "µcal/s": (4.184, "µJ/s"),
    "ucal/s": (4.184, "µJ/s"),
    "µJ": (1.0, "µJ"),
    "uJ": (1.0, "µJ"),
    "µcal": (4.184, "µJ"),
    "ucal": (4.184, "µJ"),
    "K": (lambda v: v, "K"),
    "degC": (lambda v: v + 273.15, "K"),
    "°C": (lambda v: v + 273.15, "K"),
    "C": (lambda v: v + 273.15, "K"),
    "mmol/L": (1.0, "mmol/L"),
    "mM": (1.0, "mmol/L"),
    "umol/L": (0.001, "mmol/L"),
    "µmol/L": (0.001, "mmol/L"),
    "µM": (0.001, "mmol/L"),
    "uM": (0.001, "mmol/L"),
    "mol/L": (1000.0, "mmol/L"),
}


def convert_value(raw: Any, unit: str | None, path: str = "$") -> tuple[float | None, str, dict[str, Any] | None]:
    if raw is None or raw == "":
        return None, "", {
            "severity": "error",
            "code": "MISSING_VALUE",
            "path": path,
            "message": "value is present but empty",
            "raw": raw,
        }
    unit = unit or ""
    if unit not in UNIT_ALIASES:
        raise SemanticInvalid(
            f"unsupported unit {unit!r}; raw value was retained",
            path,
            {"raw": raw, "unit": unit},
        )
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None, "", {
            "severity": "error",
            "code": "UNCONVERTIBLE_VALUE",
            "path": path,
            "message": "value cannot be converted to a number and remains in its original position",
            "raw": raw,
            "unit": unit,
        }
    if not math.isfinite(number):
        return None, "", {
            "severity": "error",
            "code": "NON_FINITE_VALUE",
            "path": path,
            "message": "numeric value is not finite and remains in its original position",
            "raw": raw,
            "unit": unit,
        }
    factor, canonical_unit = UNIT_ALIASES[unit]
    converted = factor(number) if callable(factor) else number * factor
    return float(converted), canonical_unit, None


def require_number(raw: Any, unit: str | None, path: str) -> tuple[float, str]:
    value, canonical_unit, diagnostic = convert_value(raw, unit, path)
    if diagnostic is not None or value is None:
        raise SemanticInvalid("required numeric value is unavailable", path, diagnostic or {})
    return value, canonical_unit
