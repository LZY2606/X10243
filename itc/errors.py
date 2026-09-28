from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    location: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
        }


class ItcError(Exception):
    status_code = 400

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorruptError(ItcError):
    status_code = 422


class SemanticImpossibleError(ItcError):
    status_code = 422


class ResourceBudgetExhaustedError(ItcError):
    status_code = 413


class NotFoundError(ItcError):
    status_code = 404


class ConflictError(ItcError):
    status_code = 409
