"""Three distinct domain failure classes.

``BudgetExhausted`` (resource budget spent), ``FormatCorrupt`` (malformed
input bytes/structure) and ``SemanticUnresolved`` (well-formed input with no
unique legal answer) must never collapse into one generic error.
"""

from __future__ import annotations

from typing import Any


class ItcError(Exception):
    code = "ITC_ERROR"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: list[dict[str, Any]] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostics: list[dict[str, Any]] = list(diagnostics or [])
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "diagnostics": self.diagnostics}


class BudgetExhausted(ItcError):
    code = "BUDGET_EXHAUSTED"


class FormatCorrupt(ItcError):
    code = "FORMAT_CORRUPT"


class SemanticUnresolved(ItcError):
    code = "SEMANTIC_UNRESOLVED"
