"""Distinct failure outcomes required by the domain specification.

The three outcomes must never collapse into a generic "error":

* FORMAT_CORRUPT   -- the bytes/file are not structurally parseable input.
* SEMANTIC_INVALID -- structurally valid input that has no domain meaning
                      (unconvertible value, missing correspondence, ...).
* BUDGET_EXHAUSTED -- valid input whose computation exceeds a declared
                      resource budget (sample cap, fit-evaluation cap, ...).
"""

FORMAT_CORRUPT = "FORMAT_CORRUPT"
SEMANTIC_INVALID = "SEMANTIC_INVALID"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
OK = "OK"


class ITCError(Exception):
    """Base class carrying a stable code and a position list."""

    code = SEMANTIC_INVALID

    def __init__(self, message, positions=None):
        super().__init__(message)
        self.message = message
        self.positions = list(positions or [])

    def to_diagnostic(self):
        return {
            "code": self.code,
            "message": self.message,
            "positions": self.positions,
        }


class FormatCorrupt(ITCError):
    code = FORMAT_CORRUPT


class SemanticInvalid(ITCError):
    code = SEMANTIC_INVALID


class BudgetExhausted(ITCError):
    code = BUDGET_EXHAUSTED


def diagnostic(code, message, **position):
    pos = {k: v for k, v in position.items() if v is not None}
    return {"code": code, "message": message, "positions": [pos] if pos else []}


def merge_diagnostics(*lists):
    out = []
    for lst in lists:
        out.extend(lst or [])
    return out
