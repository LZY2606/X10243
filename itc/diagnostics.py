"""Diagnostics, result statuses and the three distinct failure outcomes.

The review workflow distinguishes outcomes that must never be collapsed:

* ``FORMAT_CORRUPT``  - the payload itself is malformed and cannot be parsed;
* ``SEMANTIC_INVALID`` - the structure parsed, but it has no valid meaning
  (unknown units, missing mandatory correspondence, window crossing a gap);
* ``BUDGET_EXHAUSTED`` - the request is meaningful, but the computational
  resource budget ran out before a decision was reached.

Successful results are ``OK`` or ``STALE`` (cached value whose inputs changed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Status(str, Enum):
    OK = "ok"
    STALE = "stale"                     # cached value superseded by new inputs
    FORMAT_CORRUPT = "format_corrupt"
    SEMANTIC_INVALID = "semantic_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    location: Optional[str] = None     # e.g. "power.csv:row17", "injections#3"
    subject: Optional[str] = None      # stable identity of affected object
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "subject": self.subject,
            "detail": self.detail,
        }


class FormatCorruptError(ValueError):
    """The raw bytes/text cannot be interpreted at all."""

    def __init__(self, message: str, location: Optional[str] = None):
        super().__init__(message)
        self.location = location


class SemanticInvalidError(ValueError):
    """Parsed, but no domain-valid interpretation exists."""

    def __init__(self, message: str, diagnostics: Optional[list[Diagnostic]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class BudgetExhaustedError(RuntimeError):
    """Meaningful computation terminated by a resource budget."""

    def __init__(self, message: str, used: int = 0, budget: int = 0):
        super().__init__(message)
        self.used = used
        self.budget = budget


def err_diag(code: str, message: str, *, location: Optional[str] = None,
             subject: Optional[str] = None, detail: Optional[dict] = None) -> Diagnostic:
    return Diagnostic(code, Severity.ERROR, message, location, subject, detail or {})


def warn_diag(code: str, message: str, *, location: Optional[str] = None,
              subject: Optional[str] = None, detail: Optional[dict] = None) -> Diagnostic:
    return Diagnostic(code, Severity.WARNING, message, location, subject, detail or {})


def info_diag(code: str, message: str, *, location: Optional[str] = None,
              subject: Optional[str] = None, detail: Optional[str] = None) -> Diagnostic:
    return Diagnostic(code, Severity.INFO, message, location, subject,
                      {"detail": detail} if detail else {})
