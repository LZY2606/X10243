"""The three distinct failure outcomes required by the domain.

* :class:`FormatCorrupt`   - the bytes/text themselves cannot be parsed.
* :class:`SemanticUnresolved` - structure parses but the meaning is ambiguous
  or incomplete (missing correspondence, unknown model, ...).
* :class:`BudgetExhausted` - a deterministic compute ran past its resource
  budget before convergence; the input was fine, the answer is "not proven".
"""

from __future__ import annotations


class ItcError(Exception):
    """Base class carrying an optional machine readable ``code``."""

    code = "itc_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class FormatCorrupt(ItcError):
    code = "format_corrupt"


class SemanticUnresolved(ItcError):
    code = "semantic_unresolved"


class BudgetExhausted(ItcError):
    code = "budget_exhausted"

    def __init__(self, message: str, *, budget: int = 0, used: int = 0, code: str | None = None) -> None:
        super().__init__(message, code=code)
        self.budget = budget
        self.used = used
