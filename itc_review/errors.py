"""Distinct failure categories required by the domain contract.

The three outcome classes must never be collapsed into a generic exception:

* ResourceBudgetExhausted - a valid, well formed request used up a declared
  computational budget (fit evaluations, replay steps) before convergence.
* FormatCorrupted - the byte/text payload cannot be parsed as the declared
  format; carry the file/line/column position.
* SemanticImpossible - the input parses but the domain meaning has no legal
  solution (overlapping allocation without an overlap policy, missing blank
  correspondence, integration window bridging a data gap, etc.).

Incomplete input never returns success with nulls/defaults; callers receive a
Diagnostics collection attached to the relevant object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: str
    severity: Severity = Severity.ERROR
    value: Any | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


class Diagnostics:
    """Accumulates position-bearing diagnostics without raising."""

    def __init__(self, initial: list[Diagnostic] | None = None) -> None:
        self.items: list[Diagnostic] = list(initial or [])

    def add(
        self,
        code: str,
        message: str,
        path: str,
        severity: Severity = Severity.ERROR,
        value: Any | None = None,
        **detail: Any,
    ) -> Diagnostic:
        diag = Diagnostic(code, message, path, severity, value, detail)
        self.items.append(diag)
        return diag

    @property
    def has_error(self) -> bool:
        return any(
            d.severity in (Severity.ERROR, Severity.BLOCKED) for d in self.items
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.items]

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


class ITCError(Exception):
    """Base class for the three explicit failure categories."""


class ResourceBudgetExhausted(ITCError):
    def __init__(self, message: str, *, budget: int, used: int, **extra: Any):
        super().__init__(message)
        self.budget = budget
        self.used = used
        self.extra = extra


class FormatCorrupted(ITCError):
    def __init__(
        self,
        message: str,
        *,
        path: str,
        line: int | None = None,
        column: int | None = None,
        value: Any | None = None,
    ):
        super().__init__(message)
        self.path = path
        self.line = line
        self.column = column
        self.value = value

    def to_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            code="FORMAT_CORRUPTED",
            message=str(self),
            path=self.path,
            severity=Severity.ERROR,
            value=self.value,
            detail={"line": self.line, "column": self.column},
        )


class SemanticImpossible(ITCError):
    def __init__(self, message: str, *, path: str, code: str = "SEMANTIC_IMPOSSIBLE", **detail: Any):
        super().__init__(message)
        self.path = path
        self.semantic_code = code
        self.detail = detail

    def to_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            code=self.semantic_code,
            message=str(self),
            path=self.path,
            severity=Severity.BLOCKED,
            detail=self.detail,
        )
