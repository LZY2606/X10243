"""Units, coordinate bases and raw-representation preservation.

Every domain quantity keeps:
  * ``raw``      - exactly the representation seen at import (never mutated);
  * ``unit``     - the declared unit token;
  * ``basis``    - coordinate/time reference when relevant;
  * ``value``    - normalized numeric value in the canonical unit, or ``None``
                   when conversion is impossible.

Unconvertible values are never dropped and never defaulted: they stay at their
original position and carry a structured diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import UnconvertibleValue

# Canonical units are shown on the right.  Power is kept in µcal/s (ITC
# convention) and heat in µcal; SI alternatives are also accepted.
_UNIT_FACTOR = {
    # energy
    "ucal": 1.0,
    "µcal": 1.0,
    "mcal": 1e3,
    "cal": 1e6,
    "j": 1.0 / 4.184,
    "mj": 1.0 / 4.184 * 1e3,
    # power (energy / second)
    "ucal/s": 1.0,
    "µcal/s": 1.0,
    "mcal/s": 1e3,
    "cal/s": 1e6,
    "uw": 1.0 / 4.184,
    "µw": 1.0 / 4.184,
    "mw": 1.0 / 4.184 * 1e3,
    # volume
    "ul": 1.0,
    "µl": 1.0,
    "ml": 1e3,
    "l": 1e6,
    # amount
    "mol": 1.0,
    "mmol": 1e-3,
    "umol": 1e-6,
    "µmol": 1e-6,
    "nmol": 1e-9,
    # concentration (canonical M)
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    # time (canonical s)
    "s": 1.0,
    "sec": 1.0,
    "ms": 1e-3,
    "min": 60.0,
}

_CANONICAL = {
    "energy": "µcal",
    "power": "µcal/s",
    "volume": "µL",
    "amount": "mol",
    "concentration": "M",
    "time": "s",
}

_DIMENSION = {
    "ucal": "energy", "µcal": "energy", "mcal": "energy", "cal": "energy",
    "j": "energy", "mj": "energy",
    "ucal/s": "power", "µcal/s": "power", "mcal/s": "power", "cal/s": "power",
    "uw": "power", "µw": "power", "mw": "power",
    "ul": "volume", "µl": "volume", "ml": "volume", "l": "volume",
    "mol": "amount", "mmol": "amount", "umol": "amount", "µmol": "amount",
    "nmol": "amount",
    "m": "concentration", "mm": "concentration", "um": "concentration",
    "µm": "concentration", "nm": "concentration",
    "s": "time", "sec": "time", "ms": "time", "min": "time",
}

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Quantity:
    raw: str
    unit: str | None
    value: float | None = None
    dimension: str | None = None
    canonical_unit: str | None = None
    basis: str | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _parse_scalar(token: str) -> float | None:
    token = token.strip()
    if not token or not _NUM_RE.match(token):
        return None
    return float(token)


def parse_quantity(raw: Any, declared_unit: str | None = None,
                   dimension: str | None = None, basis: str | None = None,
                   position: dict | None = None) -> Quantity:
    """Parse a cell like ``"1.423 mL"`` or ``"0.10"`` with a declared unit."""
    raw_str = "" if raw is None else str(raw)
    token = raw_str.strip()
    num_part: str | None = None
    unit_part: str | None = None

    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]+)?$", token)
    if m:
        num_part, unit_part = m.group(1), m.group(2)
    unit = unit_part or declared_unit
    diag = []
    value = None
    norm_unit = None
    dim = dimension

    if num_part is None:
        diag.append({
            "code": "unparseable_number",
            "raw": raw_str,
            "position": position or {},
            "detail": "token is not a finite numeric literal",
        })
    elif unit is None:
        diag.append({
            "code": "missing_unit",
            "raw": raw_str,
            "position": position or {},
            "detail": "numeric value supplied without a declared unit",
        })
    else:
        key = unit.replace("μ", "µ").lower()
        factor = _UNIT_FACTOR.get(key)
        unit_dim = _DIMENSION.get(key)
        if factor is None:
            diag.append({
                "code": "unknown_unit",
                "raw": raw_str,
                "unit": unit,
                "position": position or {},
                "detail": f"unit {unit!r} has no registered conversion",
            })
        elif dimension is not None and unit_dim != dimension:
            diag.append({
                "code": "dimension_mismatch",
                "raw": raw_str,
                "unit": unit,
                "expected_dimension": dimension,
                "position": position or {},
            })
        else:
            value = float(num_part) * factor
            dim = dim or unit_dim
            norm_unit = _CANONICAL[dim]

    return Quantity(raw=raw_str, unit=unit, value=value, dimension=dim,
                    canonical_unit=norm_unit, basis=basis, diagnostics=diag)


def parse_temperature(raw: Any, declared_unit: str | None = None,
                      basis: str = "thermodynamic_temperature(K)",
                      position: dict | None = None) -> Quantity:
    raw_str = "" if raw is None else str(raw)
    token = raw_str.strip()
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]+)?$", token)
    diag = []
    value = None
    unit = None
    if not m:
        diag.append({"code": "unparseable_number", "raw": raw_str,
                     "position": position or {}})
    else:
        num = float(m.group(1))
        unit = m.group(2) or declared_unit
        if unit is None:
            diag.append({"code": "missing_unit", "raw": raw_str,
                         "position": position or {}})
        elif unit.lower() in ("c", "°c", "degc"):
            value = num + 273.15
        elif unit.lower() in ("k",):
            value = num
        else:
            diag.append({"code": "unknown_unit", "raw": raw_str, "unit": unit,
                         "position": position or {}})
    return Quantity(raw=raw_str, unit=unit, value=value, dimension="temperature",
                    canonical_unit="K", basis=basis, diagnostics=diag)


def split_number_unit(raw: Any):
    """Return (number_token, unit_token) without converting; used for samples."""
    token = "" if raw is None else str(raw).strip()
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]+)?$", token)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def require(quantity: Quantity, position: dict | None = None) -> float:
    """Return normalized value or raise UnconvertibleValue with position."""
    if quantity.value is None:
        raise UnconvertibleValue(
            "value cannot be normalized", quantity.diagnostics or
            [{"code": "unconvertible_value", "raw": quantity.raw,
              "position": position or {}}])
    return quantity.value


def diagnostic(code: str, **kwargs) -> dict[str, Any]:
    out = {"code": code}
    out.update(kwargs)
    return out
