"""Three deliberately distinct failure outcomes plus positioned diagnostics.

* FormatCorrupt        - the bytes/text cannot be parsed structurally.
* SemanticNoSolution   - structurally valid input admits no legal domain answer.
* BudgetExhausted      - search stopped because the resource budget ran out.

Diagnostics are non-fatal, positioned findings that keep the offending raw
representation in place instead of silently dropping it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = ""
    line: int | None = None
    column: int | None = None
    raw: str | None = None
    severity: str = "error"  # error | warning | info

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ITCReviewError(Exception):
    kind = "itc_error"


class FormatCorrupt(ITCReviewError):
    """Structural parse failure: framing, CSV shape, unknown section marker."""

    kind = "format_corrupt"

    def __init__(self, message: str, path: str = "", line: int | None = None,
                 column: int | None = None, raw: str | None = None):
        super().__init__(message)
        self.diagnostic = Diagnostic(
            code="FORMAT_CORRUPT", message=message, path=path,
            line=line, column=column, raw=raw)


class SemanticNoSolution(ITCReviewError):
    """Input parses but no legal domain result exists."""

    kind = "semantic_no_solution"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class BudgetExhausted(ITCReviewError):
    """Deterministic search spent its evaluation/iteration budget."""

    kind = "budget_exhausted"

    def __init__(self, message: str, spent: int = 0, budget: int = 0,
                 partial: Any = None):
        super().__init__(message)
        self.spent = spent
        self.budget = budget
        self.partial = partial


def diag_to_dict(d: Diagnostic) -> dict[str, Any]:
    return d.to_dict()
