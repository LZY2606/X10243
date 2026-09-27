"""Failure taxonomy and diagnostics.

Three distinct failure results are required by the domain rules:
  - BudgetExhausted:      a resource budget (evaluations, samples) ran out.
  - FormatCorrupt:        input bytes/structure cannot be parsed.
  - SemanticUnsolvable:   input parses but the request has no valid answer.

They are deliberately different exception types so callers (API layer,
tests, replay tooling) can distinguish the three outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class ITCFailure(Exception):
    """Base class for all domain failures."""

    kind = "failure"
    http_status = 500

    def __init__(self, message: str, position: Optional[str] = None,
                 details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.position = position
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "position": self.position,
            "details": self.details,
        }


class BudgetExhausted(ITCFailure):
    kind = "budget_exhausted"
    http_status = 507


class FormatCorrupt(ITCFailure):
    kind = "format_corrupt"
    http_status = 400


class SemanticUnsolvable(ITCFailure):
    kind = "semantic_unsolvable"
    http_status = 422


class Conflict(ITCFailure):
    kind = "conflict"
    http_status = 409


@dataclass
class Diagnostic:
    """A non-fatal finding attached to a stable position.

    Diagnostics never silently replace values: the original value stays
    where it was and the diagnostic explains why it could not be used.
    """

    code: str
    message: str
    position: Optional[str] = None
    severity: str = "warning"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagBag:
    """Collects diagnostics while a computation proceeds."""

    items: list = field(default_factory=list)

    def add(self, code: str, message: str, position: Optional[str] = None,
            severity: str = "warning") -> Diagnostic:
        d = Diagnostic(code=code, message=message, position=position,
                       severity=severity)
        self.items.append(d)
        return d

    def extend(self, others) -> None:
        for d in others:
            self.items.append(d)

    def to_list(self) -> list:
        return [d.to_dict() for d in self.items]
