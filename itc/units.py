"""Domain quantities: original representation preserved, derived normalization.

Every physical value in the project is represented by a :class:`Quantity`:

* ``raw``        - exactly what was written in the source material
* ``unit``       - the declared unit token (exact spelling retained)
* ``value``      - parsed value in SI-canonical units (``None`` if impossible)
* ``canonical_unit`` - the canonical token used downstream
* ``dimension``  - stable dimension key
* ``valid``      - parse/convert status
* ``diagnostic`` - populated when the value cannot be converted

Values that cannot be converted stay in place with a diagnostic; they are
never silently dropped, replaced by a default, or erased to make a plot draw.

Coordinate/time bases are represented by :class:`TimeBasis` and kept separate
from numeric scaling so a curve in ``t_seconds`` can never be mixed with one
in ``t_minutes`` without an explicit, recorded basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .errors import Diagnostic, diag

# dimension key -> canonical unit token
CANONICAL: dict[str, str] = {
    "time": "s",
    "power": "W",
    "energy": "J",
    "molar_energy": "J/mol",
    "volume": "L",
    "concentration": "mol/L",
    "temperature": "K",
    "dimensionless": "",
}

# unit token -> (dimension, linear factor to canonical, additive offset)
# offset is applied as canonical = raw * factor + offset
_UNITS: dict[str, tuple[str, float, float]] = {
    # time
    "s": ("time", 1.0, 0.0),
    "sec": ("time", 1.0, 0.0),
    "second": ("time", 1.0, 0.0),
    "seconds": ("time", 1.0, 0.0),
    "min": ("time", 60.0, 0.0),
    "minute": ("time", 60.0, 0.0),
    "minutes": ("time", 60.0, 0.0),
    # power
    "W": ("power", 1.0, 0.0),
    "watt": ("power", 1.0, 0.0),
    "mW": ("power", 1e-3, 0.0),
    "uW": ("power", 1e-6, 0.0),
    "µW": ("power", 1e-6, 0.0),
    # energy
    "J": ("energy", 1.0, 0.0),
    "mJ": ("energy", 1e-3, 0.0),
    "cal": ("energy", 4.184, 0.0),
    "kcal": ("energy", 4184.0, 0.0),
    # molar energy
    "J/mol": ("molar_energy", 1.0, 0.0),
    "kJ/mol": ("molar_energy", 1000.0, 0.0),
    "kcal/mol": ("molar_energy", 4184.0, 0.0),
    "cal/mol": ("molar_energy", 4.184, 0.0),
    # volume
    "L": ("volume", 1.0, 0.0),
    "mL": ("volume", 1e-3, 0.0),
    "uL": ("volume", 1e-6, 0.0),
    "µL": ("volume", 1e-6, 0.0),
    # concentration
    "mol/L": ("concentration", 1.0, 0.0),
    "M": ("concentration", 1.0, 0.0),
    "mmol/L": ("concentration", 1e-3, 0.0),
    "mM": ("concentration", 1e-3, 0.0),
    "umol/L": ("concentration", 1e-6, 0.0),
    "uM": ("concentration", 1e-6, 0.0),
    "µM": ("concentration", 1e-6, 0.0),
    "nM": ("concentration", 1e-9, 0.0),
    # temperature (offsets handled below)
    "K": ("temperature", 1.0, 0.0),
    # dimensionless
    "": ("dimensionless", 1.0, 0.0),
    "-": ("dimensionless", 1.0, 0.0),
    "fraction": ("dimensionless", 1.0, 0.0),
}


@dataclass(frozen=True)
class TimeBasis:
    """Coordinate base for a time series.

    ``origin`` is a stable label ("run_start", "injection_3", ...).  Times in
    a material are stored canonically in seconds relative to ``origin``; the
    original declared unit is retained on each quantity.
    """

    origin: str = "run_start"
    canonical_unit: str = "s"

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "canonical_unit": self.canonical_unit}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TimeBasis":
        if not data:
            return cls()
        return cls(
            origin=data.get("origin", "run_start"),
            canonical_unit=data.get("canonical_unit", "s"),
        )


@dataclass(frozen=True)
class Quantity:
    """A physical value with its original text and a normalized value."""

    raw: str
    unit: str
    dimension: str
    value: float | None = None
    canonical_unit: str = ""
    valid: bool = True
    diagnostic: Diagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "dimension": self.dimension,
            "value": self.value,
            "canonical_unit": self.canonical_unit,
            "valid": self.valid,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Quantity":
        diagnostic = data.get("diagnostic")
        return cls(
            raw=data["raw"],
            unit=data.get("unit", ""),
            dimension=data["dimension"],
            value=data.get("value"),
            canonical_unit=data.get("canonical_unit", ""),
            valid=data.get("valid", True),
            diagnostic=Diagnostic(**diagnostic) if diagnostic else None,
        )

    @property
    def required(self) -> float:
        """Return canonical value or raise if the quantity was not convertible."""
        if not self.valid or self.value is None:
            d = self.diagnostic
            from .errors import SemanticImpossible

            raise SemanticImpossible(
                f"non-convertible {self.dimension}: {self.raw!r} {self.unit!r}",
                [diag(
                    "unconvertible_value",
                    f"cannot convert {self.raw!r} {self.unit!r} to "
                    f"{CANONICAL.get(self.dimension, self.dimension)}",
                    path=d.path if d else "",
                    ref=d.ref if d else "",
                )],
            )
        return self.value


def parse_quantity(
    raw: Any,
    unit: str,
    dimension: str,
    *,
    path: str = "",
    ref: str = "",
) -> Quantity:
    """Parse ``raw`` in declared ``unit`` to the canonical unit of ``dimension``.

    The original string representation is retained verbatim.  A missing token
    or an unparseable number yields an invalid quantity with a located
    diagnostic instead of raising or substituting a default.
    """

    unit_token = (unit or "").strip()
    raw_str = "" if raw is None else str(raw).strip()
    canonical = CANONICAL[dimension]

    def invalid(code: str, message: str) -> Quantity:
        return Quantity(
            raw=raw_str,
            unit=unit_token,
            dimension=dimension,
            value=None,
            canonical_unit=canonical,
            valid=False,
            diagnostic=diag(code, message, path=path, ref=ref),
        )

    if raw is None or raw_str == "":
        return invalid("missing_value", f"missing {dimension} value")

    # Temperature offsets (only these are non-zero-origin scales).
    if dimension == "temperature":
        low = unit_token.lower()
        try:
            num = float(raw_str)
        except ValueError:
            return invalid(
                "unparseable_number",
                f"{raw_str!r} is not a numeric temperature",
            )
        if low in ("c", "°c", "degc", "celsius"):
            return Quantity(raw_str, unit_token, dimension, num + 273.15, canonical)
        if low in ("k", "kelvin"):
            return Quantity(raw_str, unit_token, dimension, num, canonical)
        if low in ("f", "°f", "degf", "fahrenheit"):
            return Quantity(
                raw_str, unit_token, dimension, (num - 32.0) * 5.0 / 9.0 + 273.15,
                canonical,
            )
        return invalid(
            "unknown_unit", f"unknown temperature unit {unit_token!r}"
        )

    spec = _UNITS.get(unit_token)
    if spec is None:
        return invalid("unknown_unit", f"unknown {dimension} unit {unit_token!r}")
    spec_dim, factor, _offset = spec
    if spec_dim != dimension:
        return invalid(
            "dimension_mismatch",
            f"unit {unit_token!r} is {spec_dim}, expected {dimension}",
        )
    try:
        num = float(raw_str)
    except ValueError:
        return invalid(
            "unparseable_number", f"{raw_str!r} is not a numeric {dimension}"
        )
    if num != num or num in (float("inf"), float("-inf")):
        return invalid("non_finite_number", f"{raw_str!r} is not finite")
    return Quantity(raw_str, unit_token, dimension, num * factor, canonical)


def quantity(value: float, dimension: str) -> Quantity:
    """Build a valid canonical quantity from an internal numeric value."""

    canonical = CANONICAL[dimension]
    return Quantity(
        raw=_format_raw(value, canonical),
        unit=canonical,
        dimension=dimension,
        value=float(value),
        canonical_unit=canonical,
    )


def _format_raw(value: float, unit: str) -> str:
    text = f"{value:.10g}"
    return f"{text} {unit}".strip()


def invalid_q(
    raw: str,
    unit: str,
    dimension: str,
    code: str,
    message: str,
    *,
    path: str = "",
    ref: str = "",
) -> Quantity:
    return Quantity(
        raw=raw,
        unit=unit,
        dimension=dimension,
        value=None,
        canonical_unit=CANONICAL[dimension],
        valid=False,
        diagnostic=diag(code, message, path=path, ref=ref),
    )


def with_value(q: Quantity, value: float) -> Quantity:
    """Return a copy carrying a derived canonical value (same raw/unit)."""

    return replace(q, value=float(value), valid=True, diagnostic=None)
