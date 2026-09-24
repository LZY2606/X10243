"""Domain primitives: stable identity, units, diagnostics, distinct result kinds."""
from __future__ import annotations

import hashlib
import json

# The three failure outcomes must stay distinguishable from each other and
# from success everywhere they surface (results, API, exports).
RESULT_OK = "ok"
RESULT_BUDGET = "budget_exhausted"
RESULT_FORMAT = "format_corrupt"
RESULT_SEMANTIC = "semantic_unsolvable"

MATERIAL_KINDS = [
    "power_trace",
    "injection_schedule",
    "concentrations",
    "temperature",
    "cell_volume",
    "baseline",
    "blank_trace",
]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def diag(code: str, message: str, path: str) -> dict:
    """A located diagnostic: never silently drop or default a value."""
    return {"code": code, "message": message, "path": path}


_OFFSET_UNITS = {("C", "K"): 273.15}
_SCALE_UNITS = {
    ("uW", "uW"): 1.0,
    ("mW", "uW"): 1000.0,
    ("uL", "uL"): 1.0,
    ("nL", "uL"): 0.001,
    ("mM", "mM"): 1.0,
    ("uM", "mM"): 0.001,
    ("mL", "mL"): 1.0,
    ("uL", "mL"): 0.001,
    ("K", "K"): 1.0,
    ("s", "s"): 1.0,
}


def convert(value, from_unit, to_unit):
    """Convert a raw value; return None when no rule exists (caller keeps raw)."""
    if value is None or from_unit is None:
        return None
    if (from_unit, to_unit) in _OFFSET_UNITS:
        return value + _OFFSET_UNITS[(from_unit, to_unit)]
    factor = _SCALE_UNITS.get((from_unit, to_unit))
    if factor is None:
        return None
    return value * factor
