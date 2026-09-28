from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ErrorKind(str, Enum):
    MALFORMED_INPUT = "malformed_input"
    SEMANTIC_UNRESOLVABLE = "semantic_unresolvable"
    RESOURCE_BUDGET_EXHAUSTED = "resource_budget_exhausted"


class ITCError(Exception):
    def __init__(self, kind: ErrorKind, message: str, diagnostics: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.diagnostics = diagnostics or []

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.kind.value,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = "$"
    severity: str = "error"
    raw_value: Any = None
    subject_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnostic(
    code: str,
    message: str,
    path: str = "$",
    severity: str = "error",
    raw_value: Any = None,
    subject_id: str | None = None,
) -> dict[str, Any]:
    return Diagnostic(code, message, path, severity, raw_value, subject_id).to_dict()


def malformed(message: str, path: str = "$", raw_value: Any = None) -> ITCError:
    return ITCError(
        ErrorKind.MALFORMED_INPUT,
        message,
        [diagnostic(ErrorKind.MALFORMED_INPUT.value, message, path, "error", raw_value)],
    )


def semantic(message: str, diagnostics: list[dict[str, Any]] | None = None) -> ITCError:
    return ITCError(ErrorKind.SEMANTIC_UNRESOLVABLE, message, diagnostics or [])

