"""Structured outcome types.

Resource budget exhaustion, corrupt format and semantic impossibility are three
distinct result classes (``BUDGET_EXHAUSTED`` / ``FORMAT_CORRUPT`` /
``SEMANTIC_IMPOSSIBLE``), never collapsed into a generic failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    path: str = ""
    value: Any = None
    raw: Any = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "value": self.value,
            "raw": self.raw,
        }


class ITCError(Exception):
    kind = "ITC_ERROR"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorrupt(ITCError):
    """The bytes/syntax cannot be parsed at all."""

    kind = "FORMAT_CORRUPT"


class SemanticImpossible(ITCError):
    """Parsed, but the request has no valid meaning."""

    kind = "SEMANTIC_IMPOSSIBLE"


class BudgetExhausted(ITCError):
    """A bounded resource (iterations/points/bytes) was consumed."""

    kind = "BUDGET_EXHAUSTED"


def as_dict(err: ITCError) -> dict:
    return {
        "kind": err.kind,
        "message": str(err),
        "diagnostics": [d.to_dict() for d in err.diagnostics],
    }
