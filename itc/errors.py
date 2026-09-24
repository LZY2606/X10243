"""Distinct failure outcomes. These three must never be conflated:

- budget_exhausted : computation ran out of its resource budget
- format_corrupt   : input bytes/structure could not be parsed
- unsolvable       : input parsed fine but is semantically unresolvable
"""

OK = "ok"
PARTIAL = "partial"  # computed, but carries diagnostics
BUDGET_EXHAUSTED = "budget_exhausted"
FORMAT_CORRUPT = "format_corrupt"
UNSOLVABLE = "unsolvable"


class ITCError(Exception):
    status = "error"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self):
        return {"status": self.status, "message": self.message,
                "diagnostics": self.diagnostics}


class BudgetExhausted(ITCError):
    status = BUDGET_EXHAUSTED


class FormatCorrupt(ITCError):
    status = FORMAT_CORRUPT


class Unsolvable(ITCError):
    status = UNSOLVABLE
