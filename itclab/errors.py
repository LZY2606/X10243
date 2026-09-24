"""Structured failure types.

The application distinguishes three non-success outcomes mandated by the
domain: resources exhausted before parsing, structurally corrupt input, and
input that parses but has no valid semantic interpretation. Each carries
position-bearing diagnostics instead of returning empty values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ITCError(Exception):
    code = "ITC_ERROR"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class BudgetExhausted(ITCError):
    """Resource budget (bytes / samples / operations) ran out."""

    code = "BUDGET_EXHAUSTED"


class FormatCorrupt(ITCError):
    """Input cannot be tokenized or parsed at all."""

    code = "FORMAT_CORRUPT"


class SemanticUnbound(ITCError):
    """Input parses but has no unique valid physical interpretation."""

    code = "SEMANTIC_UNBOUND"


class NotFound(ITCError):
    code = "NOT_FOUND"


class Conflict(ITCError):
    code = "CONFLICT"
