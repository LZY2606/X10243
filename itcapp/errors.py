"""Structured diagnostics and the three distinct failure outcomes.

Resource budget exhaustion, format corruption and semantic unsolvability
are three different results, never collapsed into one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Diagnostic:
    code: str
    severity: str  # "error" | "warning" | "info"
    location: str  # precise position: file/section/row/field
    message: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
        }


class ItcError(Exception):
    """Base error carrying structured diagnostics."""

    category = "INTERNAL"

    def __init__(self, message: str, diagnostics: Optional[List[Diagnostic]] = None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorruptedError(ItcError):
    """Input bytes/grammar are broken; nothing is imported."""

    category = "FORMAT_CORRUPTED"


class SemanticUnsolvableError(ItcError):
    """Input parses but cannot be interpreted as a valid computation request."""

    category = "SEMANTIC_UNSOLVABLE"


class BudgetExhaustedError(ItcError):
    """The configured resource budget ran out before a result was reached."""

    category = "BUDGET_EXHAUSTED"
