from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class OutcomeKind(str, Enum):
    success = "success"
    semantic_no_solution = "semantic_no_solution"
    format_corrupt = "format_corrupt"
    resource_exhausted = "resource_exhausted"


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    raw: str
    basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "raw": self.raw, "basis": self.basis}


@dataclass
class Diagnostic:
    code: str
    severity: Severity
    message: str
    location: str = ""
    value: str = ""
    expected: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value if isinstance(self.severity, Severity) else self.severity,
            "message": self.message,
            "location": self.location,
            "value": self.value,
            "expected": self.expected,
        }


@dataclass
class ParsedSeries:
    values: list[float]
    quantities: list[Quantity]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    invalid_positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "quantities": [q.to_dict() for q in self.quantities],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "invalid_positions": self.invalid_positions,
        }


@dataclass
class Gap:
    start_index: int
    end_index: int
    before_time_s: float
    after_time_s: float
    expected_interval_s: float
    actual_interval_s: float
    missing_samples: int
    source: Literal["inferred", "declared"]


@dataclass
class ParsedPower:
    time: ParsedSeries
    power: ParsedSeries
    gaps: list[Gap] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time.to_dict(),
            "power": self.power.to_dict(),
            "gaps": [g.__dict__ for g in self.gaps],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }
