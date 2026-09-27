"""Structured diagnostics.

Three *distinct* failure outcomes are modelled as separate types, never merged
into a generic "failed":

* FormatCorruptError  - bytes/JSON/schema cannot be read at all
* SemanticUnsolvable - structurally valid but the domain meaning is impossible
* BudgetExhausted     - a declared resource budget (max points / steps/time)
                        ran out before completion

Diagnostics attached to individual values carry a JSON-pointer-ish position so
the UI can highlight the original location.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

FORMAT_CORRUPT = "format_corrupt"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"
BUDGET_EXHAUSTED = "budget_exhausted"

SEVERITIES = ("error", "warning", "info")


class ImportError_(Exception):
    kind = SEMANTIC_UNSOLVABLE

    def __init__(self, message: str, *, position: Optional[str] = None,
                 detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.position = position
        self.detail = detail or {}

    def to_diagnostic(self) -> "Diagnostic":
        return Diagnostic(kind=self.kind, message=self.message,
                          position=self.position, detail=self.detail)


class FormatCorruptError(ImportError_):
    kind = FORMAT_CORRUPT


class SemanticUnsolvable(ImportError_):
    kind = SEMANTIC_UNSOLVABLE


class BudgetExhausted(ImportError_):
    kind = BUDGET_EXHAUSTED


@dataclass
class Diagnostic:
    code: str
    message: str
    kind: str = SEMANTIC_UNSOLVABLE
    severity: str = "error"
    position: Optional[str] = None
    subject: Optional[str] = None      # material/node/injection stable id
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Diagnostic":
        return Diagnostic(**d)


def diag(code: str, message: str, *, kind: str = SEMANTIC_UNSOLVABLE,
         severity: str = "error", position: Optional[str] = None,
         subject: Optional[str] = None, **detail) -> Diagnostic:
    return Diagnostic(code=code, message=message, kind=kind,
                      severity=severity, position=position,
                      subject=subject, detail=detail)


def pack(diags: list[Diagnostic]) -> list[dict]:
    return [d.to_dict() for d in diags]


def unpack(rows: list[dict]) -> list[Diagnostic]:
    return [Diagnostic.from_dict(d) for d in rows]
