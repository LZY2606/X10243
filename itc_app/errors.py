"""Three distinct failure outcomes required by the domain.

* BudgetExhausted      - computation could not finish inside the resource budget
* FormatCorrupted      - the input bytes/structure are damaged or unparsable
* SemanticNoSolution   - the input is well formed but has no unique legal meaning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ITCError(Exception):
    code = "ITC_ERROR"


class BudgetExhausted(ITCError):
    """Resource budget (e.g. function evaluations) was exhausted."""

    code = "BUDGET_EXHAUSTED"

    def __init__(self, message: str = "resource budget exhausted", used: int = 0, budget: int = 0):
        super().__init__(message)
        self.used = used
        self.budget = budget


class FormatCorrupted(ITCError):
    """Payload cannot be parsed as the declared format."""

    code = "FORMAT_CORRUPTED"

    def __init__(self, message: str = "format corrupted", position: str | None = None):
        super().__init__(message)
        self.position = position


class SemanticNoSolution(ITCError):
    """Well-formed input that admits no legal domain interpretation."""

    code = "SEMANTIC_NO_SOLUTION"

    def __init__(self, message: str = "semantic no solution", diagnostics: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


@dataclass
class Diagnostic:
    """A located diagnostic. Never replaces a value with a silent default."""

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    position: str | None = None
    value: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.position is not None:
            out["position"] = self.position
        if self.value is not None:
            out["value"] = self.value
        if self.extra:
            out["extra"] = self.extra
        return out
