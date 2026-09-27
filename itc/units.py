"""Physical quantities.

Every imported value keeps its *original representation*. A normalized value is
only attached when conversion actually succeeds; otherwise the value stays at
its original position with a diagnostic instead of being dropped or defaulted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# dimension -> { unit token -> factor to canonical base unit }
# Base units: s, J, J/s (W), mol/L (M), L, K
_PREFIX = {
    "": 1.0,
    "d": 1e-1, "c": 1e-2, "m": 1e-3,
    "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
CAL = 4.184  # exact thermochemical calorie, joule

_BASE_FACTORS: dict[str, float] = {
    # time -> s
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    # energy -> J
    "j": 1.0, "joule": 1.0, "joules": 1.0,
    "cal": CAL, "calorie": CAL,
    # power -> W = J/s
    "w": 1.0, "watt": 1.0, "watts": 1.0,
    "j/s": 1.0, "cal/s": CAL,
    # concentration -> M (mol/L)
    "m": 1.0, "mol/l": 1.0, "molar": 1.0,
    # volume -> L
    "l": 1.0, "liter": 1.0, "litre": 1.0, "liters": 1.0, "litres": 1.0,
    # temperature base handled specially -> K
    "k": float("nan"),
}

_DIMENSIONS = {
    "time": ("s", {"s", "sec", "secs", "second", "seconds", "min",
                   "minute", "minutes", "ms"}),
    "energy": ("J", {"j", "joule", "joules", "cal", "calorie",
                     "mj", "uj", "µj", "μj", "nj", "mcal", "ucal",
                     "µcal", "μcal", "kcal", "kj"}),
    "power": ("W", {"w", "watt", "watts", "j/s", "mw", "uw", "µw",
                    "μw", "nw", "cal/s", "ucal/s", "µcal/s", "μcal/s",
                    "mcal/s"}),
    "concentration": ("M", {"m", "mol/l", "molar", "mm", "um", "µm",
                            "μm", "nm", "pm"}),
    "volume": ("L", {"l", "liter", "litre", "liters", "litres", "ml",
                     "ul", "µl", "μl", "nl", "pl"}),
    "temperature": ("K", {"k", "c", "°c", "ºc", "degc", "f", "°f",
                          "degf"}),
}


def _split_prefix(unit: str) -> tuple[float, str]:
    u = unit.strip().lower().replace(" ", "")
    if u in _BASE_FACTORS or "/" in u or u in {"°c", "ºc", "°f"}:
        return 1.0, u
    if u and u[0] in _PREFIX:
        return _PREFIX[u[0]], u
    return 1.0, u


def canonical_unit(dimension: str) -> str:
    return _DIMENSIONS[dimension][0]


@dataclass(frozen=True)
class UnitInfo:
    dimension: str
    canonical: str
    factor: float  # multiply value by factor to reach canonical (Celsius/f handled)
    affine: bool = False  # temperature offset


def classify(unit: Optional[str]) -> Optional[UnitInfo]:
    """Resolve a unit string to its dimension and conversion factor."""
    if unit is None:
        return None
    raw = str(unit).strip()
    key = raw.lower().replace(" ", "")
    for dim, (base, tokens) in _DIMENSIONS.items():
        if key in tokens:
            if dim == "temperature":
                if key in {"c", "°c", "ºc", "degc"}:
                    return UnitInfo(dim, "K", 1.0, affine=True)
                if key in {"f", "°f", "degf"}:
                    return UnitInfo(dim, "K", 5.0 / 9.0, affine=True)
                return UnitInfo(dim, "K", 1.0, affine=False)
            # compound per-second power units
            if "/" in key:
                top = key.split("/", 1)[0]
                p, base_tok = _split_prefix(top)
                if base_tok in {"cal", "calorie"}:
                    return UnitInfo(dim, base, p * CAL)
                return UnitInfo(dim, base, p)
            p, base_tok = _split_prefix(key)
            if base_tok in {"cal", "calorie"}:
                return UnitInfo(dim, base, p * CAL)
            return UnitInfo(dim, base, p)
    return None


def _to_float(text: Any) -> Optional[float]:
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(float(text)) else None
    s = str(text).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def convert(value: Any, unit: Optional[str], dimension: str):
    """Return (normalized_value, canonical_unit, error_or_none).

    Never raises: an unconvertible value yields (None, canonical, reason) and
    the caller must keep the raw text at its original position.
    """
    base = canonical_unit(dimension)
    v = _to_float(value)
    if v is None:
        return None, base, f"not-a-number: {value!r}"
    info = classify(unit)
    if info is None:
        return None, base, f"unknown {dimension} unit: {unit!r}"
    if info.dimension != dimension:
        return None, base, f"unit {unit!r} is {info.dimension}, expected {dimension}"
    if info.affine:
        key = str(unit).strip().lower().replace(" ", "")
        if key in {"f", "°f", "degf"}:
            return (v - 32.0) * 5.0 / 9.0 + 273.15, base, None
        return v + 273.15, base, None
    return v * info.factor, base, None


@dataclass(frozen=True)
class Quantity:
    """An imported measurement: raw text + raw unit + normalized value."""
    raw: str
    raw_unit: Optional[str]
    dimension: str
    value: Optional[float]          # in canonical unit; None if unconverted
    unit: str                       # canonical unit
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error is None

    def display(self) -> str:
        if self.ok:
            return f"{_fmt(self.value)} {self.unit}"
        return f"{self.raw} {self.raw_unit or ''}".strip() + " (unconverted)"


def _fmt(v: float) -> str:
    av = abs(v)
    if av != 0 and (av < 1e-4 or av >= 1e7):
        return f"{v:.6g}"
    return f"{v:.6g}"


def quantity(value: Any, unit: Optional[str], dimension: str) -> Quantity:
    norm, base, err = convert(value, unit, dimension)
    return Quantity(raw="" if value is None else str(value),
                    raw_unit=unit, dimension=dimension, value=norm,
                    unit=base, error=err)


def all_ok(*qs: Quantity) -> bool:
    return all(q.ok for q in qs)
