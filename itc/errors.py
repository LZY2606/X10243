"""Three distinct failure classes required by the domain specification.

* ResourceExhausted - the operation hit an explicit resource budget (rows,
  points, fit evaluations). Input and format may be perfectly valid.
* FormatCorrupted   - the bytes/text cannot be parsed structurally; a
  position (file/line/column) is always attached when available.
* SemanticImpasse   - structurally valid input with no legal interpretation
  (incomplete data, unconvertible values, endpoint outside the trace, ...).

Each carries *positioned* diagnostics; callers must not substitute defaults
or nulls for a failed value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: str = "error"  # error | warning
    file: str | None = None
    line: int | None = None
    column: int | None = None
    section: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "section": self.section,
            "raw": self.raw,
        }


class ItcError(Exception):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: list[Diagnostic] = diagnostics or []


class ResourceExhausted(ItcError):
    """Budget exceeded - distinct from corruption and semantic failure."""

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message, diagnostics)


class FormatCorrupted(ItcError):
    """Structural parse failure, always with a position when one exists."""

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message, diagnostics)


class SemanticImpasse(ItcError):
    """Parsed input that has no legal physical interpretation."""

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message, diagnostics)
