"""Three distinct failure outcomes required by the specification.

* FormatCorrupt      - the payload is malformed / unreadable (syntax-level).
* SemanticUnsolvable - the payload is readable but has no valid meaning.
* BudgetExhausted    - an iterative computation used its resource budget
                       before converging; the input was valid, the result is
                       provisional and carries the count of iterations used.

All three carry machine-readable location/position information and never use
None or a silent default to pretend success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ITCError(Exception):
    code = "itc_error"
    http_status = 400

    def __init__(self, message: str, *, locations: list[dict[str, Any]] | None = None,
                 diagnostics: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.locations = locations or []
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "locations": self.locations,
            "diagnostics": self.diagnostics,
        }


class FormatCorrupt(ITCError):
    code = "format_corrupt"
    http_status = 422


class SemanticUnsolvable(ITCError):
    code = "semantic_unsolvable"
    http_status = 422


class BudgetExhausted(ITCError):
    code = "budget_exhausted"
    http_status = 409

    def __init__(self, message: str, *, budget: int, used: int,
                 locations: list[dict[str, Any]] | None = None):
        super().__init__(message, locations=locations,
                         diagnostics=[{"budget": budget, "used": used}])
        self.budget = budget
        self.used = used


@dataclass
class Diagnostic:
    """A value that could not be converted: left in place, never dropped."""
    severity: str            # "error" | "warning"
    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "raw": self.raw,
        }
