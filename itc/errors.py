"""Typed failure results.

The three failure classes demanded by the specification are intentionally
distinct so callers (and tests) can tell a corrupted input file apart from a
semantically impossible request and from an exhausted numerical budget:

* :class:`FormatCorrupted`  - bytes/text that cannot be parsed at all
* :class:`SemanticImpossible` - parseable input that contradicts the domain
* :class:`BudgetExhausted` - a numerical search used its entire iteration budget

All structured failures carry ``diagnostics``: a list of
``{code, message, path, ref}`` objects with explicit positions.  Diagnostics
never use ``None`` placeholders for missing required data; an incomplete input
is a failure, not a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = ""
    ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "ref": self.ref,
        }


class ITCError(Exception):
    """Base class carrying structured, position-aware diagnostics."""

    kind = "error"

    def __init__(
        self,
        message: str,
        diagnostics: list[Diagnostic] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostics: list[Diagnostic] = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupted(ITCError):
    """The input cannot be decoded/parsed structurally."""

    kind = "format_corrupted"


class SemanticImpossible(ITCError):
    """Structurally valid input that is meaningless for ITC review."""

    kind = "semantic_impossible"


class BudgetExhausted(ITCError):
    """A numerical process consumed its resource budget without converging."""

    kind = "budget_exhausted"


def diag(code: str, message: str, path: str = "", ref: str = "") -> Diagnostic:
    return Diagnostic(code=code, message=message, path=path, ref=ref)


@dataclass
class Failure:
    """A non-raising carrier for a structured failure (used by derived results)."""

    kind: str
    message: str
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @classmethod
    def from_error(cls, exc: ITCError) -> "Failure":
        return cls(kind=exc.kind, message=exc.message, diagnostics=list(exc.diagnostics))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Failure":
        return cls(
            kind=data["kind"],
            message=data["message"],
            diagnostics=[Diagnostic(**d) for d in data.get("diagnostics", [])],
        )
