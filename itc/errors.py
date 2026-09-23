"""Three strictly distinct failure outcomes.

* FormatCorrupt       - the bundle is structurally unreadable.
* SemanticImpossible  - the bundle parses but cannot be interpreted as ITC input.
* BudgetExhausted     - a declared resource budget ran out before parsing finished.

Value-level problems (an individual value that cannot be converted) are not hard
errors: the raw representation is kept in place and a diagnostic is attached.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    location: str
    severity: str = "error"  # error | warning | info
    raw: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Diagnostic":
        return cls(
            code=data["code"],
            message=data["message"],
            location=data["location"],
            severity=data.get("severity", "error"),
            raw=data.get("raw"),
        )


class ITCError(Exception):
    code = "itc_error"


class FormatCorrupt(ITCError):
    """Structure is broken: unclosed section, no material header, bad CSV shape."""

    code = "format_corrupt"

    def __init__(self, message: str, location: str = "document", raw: str | None = None):
        super().__init__(message)
        self.diagnostic = Diagnostic(self.code, message, location, "error", raw)


class SemanticImpossible(ITCError):
    """Structure parses but required domain content is missing or contradictory."""

    code = "semantic_impossible"

    def __init__(self, message: str, location: str = "document", raw: str | None = None):
        super().__init__(message)
        self.diagnostic = Diagnostic(self.code, message, location, "error", raw)


class BudgetExhausted(ITCError):
    """A declared resource budget (samples/sections/bytes) was exceeded."""

    code = "budget_exhausted"

    def __init__(self, message: str, location: str = "document", raw: str | None = None):
        super().__init__(message)
        self.diagnostic = Diagnostic(self.code, message, location, "error", raw)
