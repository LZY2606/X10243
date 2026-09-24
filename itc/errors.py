from __future__ import annotations

from typing import Any


class ITCError(Exception):
    kind = "ITC_ERROR"
    status_code = 400

    def __init__(self, message: str, path: str = "$", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }


class FormatCorrupt(ITCError):
    kind = "FORMAT_CORRUPT"
    status_code = 422


class SemanticInvalid(ITCError):
    kind = "SEMANTIC_INVALID"
    status_code = 422


class ResourceExhausted(ITCError):
    kind = "RESOURCE_EXHAUSTED"
    status_code = 413


class ComputationInfeasible(ITCError):
    kind = "SEMANTIC_NO_SOLUTION"
    status_code = 422
