"""Three deliberately distinct failure outcomes.

* :class:`ResourceBudgetExhausted` - the input was valid but the configured
  processing budget (e.g. maximum samples) ran out.
* :class:`FormatCorrupt`      - the bytes/text cannot be parsed at all.
* :class:`SemanticUnresolvable` - structurally valid input that cannot be given
  a unique/valid physical meaning (window crosses a data gap, blank has no
  corresponding injection, ...).

Each failure carries position-tagged diagnostics; callers must never substitute
nulls/defaults to fake success.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    location: str = ""                 # e.g. "row 12 col time", "injection i=3"
    severity: str = "error"            # error | warning
    raw: str | None = None             # original representation, retained
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ITCError(Exception):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.__class__.__name__,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class ResourceBudgetExhausted(ITCError):
    """Valid input, but a finite resource budget was exceeded."""


class FormatCorrupt(ITCError):
    """Bytes/text that cannot be parsed structurally."""


class SemanticUnresolvable(ITCError):
    """Parsable, but no unique/valid physical interpretation exists."""


class StaleReference(SemanticUnresolvable):
    """A result references a parent version that is no longer current."""


def diag(code: str, message: str, location: str = "", *, severity: str = "error",
         raw: str | None = None, suggestion: str | None = None) -> Diagnostic:
    return Diagnostic(code, message, location, severity, raw, suggestion)
