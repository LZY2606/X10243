"""Quantities, units and coordinate bases.

Raw material representations are preserved verbatim by the storage layer;
this module is the *derived* normalization layer.  Every converted value is
returned together with its unit and dimension.  Values that cannot be
converted produce a :class:`QuantityError` carrying a machine-readable
diagnostic with a ``where`` location instead of being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class QuantityError(ValueError):
    """A value that cannot be converted.

    ``diagnostic`` is a machine-readable dict (code/where/message/raw) that
    callers attach to events.  This is the *semantic impossibility* channel,
    distinct from corrupt formatting (``ValueError`` from parsers) and from
    exhausted compute budgets (``BudgetExhausted``).
    """

    def __init__(self, code: str, where: str, message: str, raw: Any = None):
        super().__init__(message)
        self.diagnostic = {"code": code, "where": where, "message": message, "raw": raw}


@dataclass(frozen=True)
class Quantity:
    """A scalar in canonical units."""

    value: float
    unit: str
    dim: str
    raw: Any = None
    raw_unit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "dim": self.dim,
            "raw": self.raw,
            "raw_unit": self.raw_unit,
        }


# Canonical dimensions -> canonical unit, with multiplicative aliases.
_UNITS: dict[str, dict[str, float]] = {
    "power": {
        "uW": 1.0,
        "µW": 1.0,
        "microwatt": 1.0,
        "microwatts": 1.0,
        "mW": 1000.0,
        "milliwatt": 1000.0,
    },
    "time": {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "min": 60.0,
        "minute": 60.0,
    },
    "volume": {
        "uL": 1.0,
        "µL": 1.0,
        "ul": 1.0,
        "microliter": 1.0,
        "microlitre": 1.0,
        "mL": 1000.0,
        "ml": 1000.0,
        "milliliter": 1000.0,
        "L": 1_000_000.0,
    },
    "concentration": {
        "mM": 1.0,
        "mmol/L": 1.0,
        "mmol/l": 1.0,
        "uM": 1e-3,
        "µM": 1e-3,
        "micromolar": 1e-3,
        "nM": 1e-6,
        "nanomolar": 1e-6,
        "M": 1000.0,
        "mol/L": 1000.0,
    },
    "heat": {
        "uJ": 1.0,
        "µJ": 1.0,
        "microjoule": 1.0,
        "mJ": 1000.0,
        "mcal": 4.184,
        "cal": 4184.0,
        "J": 1_000_000.0,
    },
    "heat_per_mole": {
        "kcal/mol": 1000.0,
        "cal/mol": 1.0,
        "kJ/mol": 239.005736,
    },
}

_DIM_NAMES = {
    "power": "power",
    "time": "time",
    "volume": "volume",
    "concentration": "concentration",
    "temperature": "temperature",
    "heat": "heat",
    "heat_per_mole": "molar heat",
}


def dims() -> list[str]:
    return ["power", "time", "volume", "concentration", "temperature", "heat", "heat_per_mole"]


def _to_float(raw: Any, where: str) -> float:
    try:
        out = float(raw)
    except (TypeError, ValueError) as exc:
        raise QuantityError(
            "not-a-number", where, f"value {raw!r} is not numeric", raw=raw
        ) from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise QuantityError("not-finite", where, f"value {raw!r} is not finite", raw=raw)
    return out


def parse_quantity(raw: Any, dim: str, where: str) -> Quantity:
    """Normalize a scalar measurement to canonical units.

    Accepts a number, ``"12.3 uM"``, or ``{"value": 12.3, "unit": "uM"}``.
    Temperature is handled by :func:`parse_temperature`.
    """
    if dim == "temperature":
        return parse_temperature(raw, where)
    table = _UNITS.get(dim)
    if table is None:
        raise QuantityError("unknown-dimension", where, f"unknown dimension {dim!r}", raw=raw)
    if isinstance(raw, dict):
        if "value" not in raw or "unit" not in raw:
            raise QuantityError(
                "missing-unit",
                where,
                "quantity object requires 'value' and 'unit'",
                raw=raw,
            )
        value_raw, unit = raw["value"], str(raw["unit"])
    elif isinstance(raw, str):
        text = raw.strip()
        head = text.rstrip()
        cut = 0
        for ch in reversed(head):
            if ch.isalpha() or ch in ("µ", "/"):
                cut += 1
            else:
                break
        cut = len(head) - cut
        value_part, unit = head[:cut].strip(), head[cut:].strip()
        if not value_part or not unit:
            raise QuantityError(
                "missing-unit",
                where,
                f"cannot split value and unit from {raw!r}",
                raw=raw,
            )
        value_raw = value_part
    else:
        raise QuantityError(
            "unrepresentable",
            where,
            "scalar must be a number, string or {value, unit} object",
            raw=raw,
        )
    value = _to_float(value_raw, where)
    if unit not in table:
        raise QuantityError(
            "unknown-unit",
            where,
            f"unit {unit!r} is not a {_DIM_NAMES.get(dim, dim)} unit",
            raw=raw,
        )
    canonical = value * table[unit]
    canonical_unit = {
        "power": "uW",
        "time": "s",
        "volume": "uL",
        "concentration": "mM",
        "heat": "uJ",
        "heat_per_mole": "cal/mol",
    }[dim]
    return Quantity(canonical, canonical_unit, dim, raw=raw, raw_unit=unit)


def parse_temperature(raw: Any, where: str) -> Quantity:
    """Normalize temperature to kelvin, retaining the original representation."""
    if isinstance(raw, dict):
        if "value" not in raw or "unit" not in raw:
            raise QuantityError(
                "missing-unit", where, "temperature requires value and unit", raw=raw
            )
        value = _to_float(raw["value"], where)
        unit = str(raw["unit"]).upper()
        original = raw
    elif isinstance(raw, str):
        text = raw.strip()
        for suffix in ("°C", "C", "°F", "F", "K"):
            if text.endswith(suffix):
                unit = suffix.upper().replace("°", "")
                value = _to_float(text[: -len(suffix)].strip(), where)
                original = raw
                break
        else:
            raise QuantityError("unknown-unit", where, f"temperature {raw!r} lacks C/F/K", raw=raw)
    else:
        raise QuantityError("unrepresentable", where, "temperature needs a unit", raw=raw)
    if unit == "K":
        kelvin = value
    elif unit == "C":
        kelvin = value + 273.15
    elif unit == "F":
        kelvin = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:
        raise QuantityError("unknown-unit", where, f"temperature unit {unit!r}", raw=raw)
    return Quantity(kelvin, "K", "temperature", raw=original, raw_unit=unit)


def parse_array(points: Any, dim: str, where: str) -> tuple[list[float], list[dict[str, Any]]]:
    """Normalize a series; bad points stay *in place* with diagnostics.

    Returns canonical values with ``None`` holes and a diagnostics list.
    Points are never dropped just so a plot can be drawn.
    """
    if not isinstance(points, list) or not points:
        raise QuantityError(
            "empty-or-malformed-array", where, "expected a non-empty list", raw=points
        )
    values: list[float | None] = []
    diagnostics: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        try:
            values.append(parse_quantity(point, dim, f"{where}[{idx}]").value)
        except QuantityError as exc:
            values.append(None)
            exc.diagnostic["index"] = idx
            diagnostics.append(exc.diagnostic)
    return values, diagnostics
