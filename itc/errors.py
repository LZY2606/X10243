"""Three distinct failure outcomes required by the domain spec.

- ITCFormatError       -> resource/format damage (unparseable bytes/CSV)
- ITCSemanticError     -> semantically unresolvable (contradictory domain state)
- BudgetExhaustedError -> resource budget exhausted (sample/iteration caps)

Every error carries machine-readable positional diagnostics, never silent
defaults.
"""
from __future__ import annotations


class ITCError(Exception):
    def __init__(self, message: str, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = list(diagnostics or [])

    def to_event(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


class ITCFormatError(ITCError):
    """The payload itself is damaged: unknown encoding, malformed CSV/JSON,
    missing required columns, unparseable quantity token."""

    code = "FORMAT_CORRUPT"


class ITCSemanticError(ITCError):
    """Payload is well-formed but the request has no legal domain answer:
    negative time, window that cannot avoid a gap, duplicate injection ids,
    contradictory temperatures, ..."""

    code = "SEMANTIC_UNRESOLVABLE"


class BudgetExhaustedError(ITCError):
    """The request is valid but exceeds a declared resource budget
    (max samples, max fit evaluations, max zip members)."""

    code = "BUDGET_EXHAUSTED"


def diag(location: str, code: str, message: str, value=None, **extra) -> dict:
    d = {"location": location, "code": code, "message": message}
    if value is not None:
        d["value"] = value
    d.update(extra)
    return d
