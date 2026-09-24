"""Distinct failure outcomes required by the specification.

Three outcome families must never be collapsed into one generic error:

* :class:`FormatCorrupt`  - the bytes/structure of an input are malformed,
* :class:`SemanticInfeasible` - structure parses but domain rules reject it,
* :class:`BudgetExhausted` - a bounded computation ran out of its resource
  allowance before a result could be produced.

Every diagnostic carries a ``location`` (JSON-pointer-ish path into the input)
so callers can show the operator exactly which fragment was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    location: str = ""
    value: Any = None
    severity: str = "error"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "value": self.value,
            "severity": self.severity,
        }


class ITCError(Exception):
    """Base error carrying structured diagnostics."""

    outcome = "unknown"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: list[Diagnostic] = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupt(ITCError):
    """The input representation is broken (bad JSON, bad CSV, bad checksum...)."""

    outcome = "format_corrupt"


class SemanticInfeasible(ITCError):
    """The input parses but violates a domain rule.

    Examples: injection time outside the sampled interval, duplicate injection
    ordinal, concentration revision whose value cannot be converted.
    """

    outcome = "semantic_infeasible"


class BudgetExhausted(ITCError):
    """A bounded search/evaluation used all of its resource budget."""

    outcome = "budget_exhausted"


def raise_for_diagnostics(diagnostics: list[Diagnostic], what: str) -> None:
    hard = [d for d in diagnostics if d.severity == "error"]
    if hard:
        raise SemanticInfeasible(f"{what} rejected: {hard[0].message}", hard)
