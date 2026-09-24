"""Distinct, testable failure classes required by the ITC review workflow.

Resource budget exhaustion, corrupt format and semantic impossibility must be
three different results rather than a single generic exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ITCError(Exception):
    """Base class for failures that carry a machine-readable kind."""

    kind = "itc_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CorruptFormat(ITCError):
    """The bytes/rows cannot be parsed as the claimed artifact."""

    kind = "corrupt_format"


class SemanticImpossibility(ITCError):
    """Parsed inputs exist but cannot mean what they claim
    (negative time steps, unknown units, inconsistent references...)."""

    kind = "semantic_impossibility"


class BudgetExhausted(ITCError):
    """A bounded computation (model fit) spent its evaluation budget."""

    kind = "budget_exhausted"

    def __init__(self, message: str, budget: int, used: int, partial=None):
        super().__init__(message)
        self.budget = budget
        self.used = used
        self.partial = partial


class IdentityConflict(ITCError):
    """Import collided with an existing immutable identity."""

    kind = "identity_conflict"


@dataclass
class Diagnostic:
    """A located, structured problem that does not necessarily abort the run.

    Every value carries its location (file/json path + index), a stable code,
    severity and the original representation so nothing is dropped for display.
    """

    code: str
    message: str
    severity: str = "error"  # error | warning | info
    location: str = ""
    original: object = None
    fatal: bool = False
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "original": self.original,
            "fatal": self.fatal,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnostic":
        return cls(
            code=d["code"],
            message=d["message"],
            severity=d.get("severity", "error"),
            location=d.get("location", ""),
            original=d.get("original"),
            fatal=d.get("fatal", False),
            context=d.get("context", {}),
        )


def raise_for_fatal(diagnostics: list[Diagnostic]) -> None:
    """Convert the first fatal diagnostic into the matching typed error."""
    for diag in diagnostics:
        if diag.fatal:
            if diag.code.startswith("format"):
                raise CorruptFormat(diag.message)
            raise SemanticImpossibility(diag.message)
