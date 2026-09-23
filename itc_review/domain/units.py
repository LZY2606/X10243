"""Quantities keep their original textual representation.

A ``Quantity`` carries ``raw`` (exactly what was imported), the parsed SI value
used by the derivation layer, the canonical unit and the coordinate/reference
basis.  Values that cannot be converted stay at the origin with a diagnostic;
they are never replaced by a default just to let a plot render.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .errors import Diagnostic

# canonical unit -> (set of accepted aliases, linear scale to canonical, offset)
_SCALE: dict[str, dict[str, float]] = {
    "s": {"sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
          "min": 60.0, "minute": 60.0, "minutes": 60.0},
    "uW": {"microwatt": 1.0, "µW": 1.0, "uW": 1.0,
           "mW": 1000.0, "W": 1_000_000.0},
    "uJ": {"µJ": 1.0, "uJ": 1.0, "microjoule": 1.0,
           "mJ": 1000.0, "J": 1_000_000.0},
    "kcal/mol": {"kcal/mol": 1.0, "kcal/mole": 1.0,
                 "cal/mol": 0.001, "J/mol": 1.0 / 4184.0,
                 "kJ/mol": 1.0 / 4.184},
    "uL": {"µL": 1.0, "uL": 1.0, "microliter": 1.0, "microlitre": 1.0,
           "mL": 1000.0, "ml": 1000.0, "L": 1_000_000.0},
    "mM": {"mmol/L": 1.0, "mm": 1.0, "um": 0.001, "µM": 0.001, "uM": 0.001,
           "nm": 1e-6, "M": 1000.0, "mol/L": 1000.0},
    "K": {"k": 1.0, "kelvin": 1.0},
    "1/M": {"M^-1": 1.0, "M-1": 1.0, "/M": 1.0, "1/M": 1.0},
    "": {"": 1.0, "dimensionless": 1.0, "-": 1.0, "unitless": 1.0},
}

# temperature aliases need offset handling
_CELSIUS = {"c": "K", "degc": "K", "°c": "K", "celsius": "K"}


@dataclass
class Quantity:
    raw: str
    value: float | None          # canonical SI value; None when unconvertible
    unit: str                    # canonical unit ("" for dimensionless)
    basis: str = ""              # coordinate/reference basis, free text
    ok: bool = True
    diagnostic: Diagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.diagnostic is not None:
            d["diagnostic"] = self.diagnostic.to_dict()
        return d


def _number(token: str) -> float:
    return float(token.strip())


def parse_quantity(raw: Any, canonical: str | None = None,
                   basis: str = "", path: str = "",
                   line: int | None = None) -> Quantity:
    """Parse ``"<number> <unit>"`` preserving the raw text.

    If ``canonical`` is given the unit may be omitted and is assumed canonical.
    Unconvertible input returns ``ok=False`` with a positioned diagnostic and
    ``value=None``; the caller must keep the raw token.
    """
    original = "" if raw is None else str(raw).strip()
    if original == "":
        return Quantity(raw=original, value=None, unit=canonical or "",
                        basis=basis, ok=False,
                        diagnostic=Diagnostic(
                            code="EMPTY_VALUE",
                            message="empty value cannot be converted",
                            path=path, line=line, raw=original))
    parts = original.split(None, 1)
    try:
        number = _number(parts[0])
    except ValueError:
        return Quantity(raw=original, value=None, unit=canonical or "",
                        basis=basis, ok=False,
                        diagnostic=Diagnostic(
                            code="UNCONVERTIBLE_VALUE",
                            message=f"cannot parse number from {original!r}",
                            path=path, line=line, raw=original))
    unit_token = parts[1].strip() if len(parts) > 1 else (canonical or "")
    low = unit_token.replace("μ", "µ")

    # temperature with offset
    if low in _CELSIUS:
        return Quantity(raw=original, value=number + 273.15, unit="K",
                        basis=basis)
    if low == "f" or low == "°f":
        return Quantity(raw=original, value=(number - 32.0) * 5.0 / 9.0
                        + 273.15, unit="K", basis=basis)

    for canon, aliases in _SCALE.items():
        if canonical is not None and canon != canonical:
            continue
        if low.lower() in {a.lower() for a in aliases}:
            return Quantity(raw=original, value=number * aliases[
                next(k for k in aliases if k.lower() == low.lower())],
                unit=canon, basis=basis)
    # when no canonical requested, try every table
    if canonical is None:
        for canon, aliases in _SCALE.items():
            if low.lower() in {a.lower() for a in aliases}:
                key = next(k for k in aliases if k.lower() == low.lower())
                return Quantity(raw=original, value=number * aliases[key],
                                unit=canon, basis=basis)

    return Quantity(raw=original, value=None, unit=canonical or unit_token,
                    basis=basis, ok=False,
                    diagnostic=Diagnostic(
                        code="UNKNOWN_UNIT",
                        message=f"unit {unit_token!r} cannot be converted"
                                f" to {canonical or 'any canonical unit'}",
                        path=path, line=line, raw=original))
