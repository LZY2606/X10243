"""Distinct failure kinds required by the domain.

* :class:`FormatCorrupted` - the input bytes/structure cannot be parsed.
* :class:`SemanticUnresolvable` - the input parses but has no valid domain
  meaning (unknown unit, inconsistent references, ...).
* :class:`BudgetExhausted` - a computation hit a declared resource budget
  before converging.

Every diagnostic carries a ``location`` (JSON-pointer-like path into the
material that owns it) so an incomplete import never has to be disguised with
nulls or defaults.
"""
from __future__ import annotations

from typing import Any


class ITCError(Exception):
    kind = "itc_error"

    def __init__(self, message: str, *, location: str = "", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out = {"kind": self.kind, "message": self.message, "location": self.location}
        if self.details:
            out["details"] = self.details
        return out


class FormatCorrupted(ITCError):
    """The payload itself is malformed (bad JSON, wrong section shape)."""

    kind = "format_corrupted"


class SemanticUnresolvable(ITCError):
    """Parsed payload, but the domain semantics cannot be satisfied."""

    kind = "semantic_unresolvable"


class BudgetExhausted(ITCError):
    """A deterministic resource budget ran out before convergence."""

    kind = "budget_exhausted"


class NotFound(ITCError):
    kind = "not_found"


class Conflict(ITCError):
    """A decision conflicts with an existing decision for the same scope."""

    kind = "conflict"


class Diagnostic:
    """A non-fatal diagnostic attached to a value kept in place."""

    __slots__ = ("code", "message", "location", "severity")

    def __init__(self, code: str, message: str, location: str = "", severity: str = "warning"):
        self.code = code
        self.message = message
        self.location = location
        self.severity = severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
        }
