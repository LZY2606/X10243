"""Three deliberately distinct failure results.

* :class:`FormatCorrupt`     - the bytes/structure cannot be parsed at all.
* :class:`SemanticUnsolvable` - structurally valid but no legal interpretation.
* :class:`BudgetExhausted`   - the evaluation budget ran out before completion.

Callers (tests included) can distinguish them by class or by ``code``.
"""
from __future__ import annotations


class ITCPipelineError(Exception):
    code = "itc_error"


class FormatCorrupt(ITCPipelineError):
    code = "format_corrupt"

    def __init__(self, message: str, *, where: str | None = None):
        super().__init__(message)
        self.where = where


class SemanticUnsolvable(ITCPipelineError):
    code = "semantic_unsolvable"

    def __init__(self, message: str, *, diagnostics: list | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class BudgetExhausted(ITCPipelineError):
    code = "budget_exhausted"

    def __init__(self, message: str, *, used: int = 0, limit: int = 0):
        super().__init__(message)
        self.used = used
        self.limit = limit


class NotFound(KeyError):
    pass
