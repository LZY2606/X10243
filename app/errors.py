"""Distinct failure outcomes for ITC import/compute workflows.

The three mandated outcomes are deliberately separate exception types so
callers (and the storage layer) can distinguish format corruption, semantic
impossibility and resource-budget exhaustion.
"""


class ITCError(Exception):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self):
        return {"kind": self.kind, "message": self.message, "diagnostics": self.diagnostics}


class FormatCorrupt(ITCError):
    kind = "format_corrupt"


class SemanticUnsolvable(ITCError):
    kind = "semantic_unsolvable"


class IncompleteInput(ITCError):
    kind = "incomplete_input"


class BudgetExhausted(ITCError):
    kind = "budget_exhausted"


class UnconvertibleValue(ITCError):
    kind = "unconvertible_value"
