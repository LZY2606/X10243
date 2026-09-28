"""The three result classes required by the domain contract.

* FormatCorrupt      - the input bytes/text cannot be read as the declared
                       structure at all (damaged file, missing section, ...).
* SemanticUnsolvable - the structure parses, but domain rules give no valid
                       answer (e.g. an integration window lands inside a data
                       gap, a concentration cannot be converted, ...).
* BudgetExhausted    - the computation was stopped by a resource cap (too
                       many samples, fit evaluation budget used up, ...).

They stay separate so callers and tests can distinguish them; they are never
collapsed into a generic failure.
"""
from __future__ import annotations


class ITCError(Exception):
    code = "ITC_ERROR"


class FormatCorrupt(ITCError):
    code = "FORMAT_CORRUPT"

    def __init__(self, message: str, location: str = "", raw: str | None = None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.raw = raw


class SemanticUnsolvable(ITCError):
    code = "SEMANTIC_UNSOLVABLE"

    def __init__(self, message: str, location: str = "", diagnostics=None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.diagnostics = diagnostics or []


class BudgetExhausted(ITCError):
    code = "BUDGET_EXHAUSTED"

    def __init__(self, message: str, budget: str = "", used=None, limit=None):
        super().__init__(message)
        self.message = message
        self.budget = budget
        self.used = used
        self.limit = limit
