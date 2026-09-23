"""Structured result statuses and diagnostics.

Three failure modes are deliberately distinct result statuses, never
collapsed into one another and never masked by null/default success:

- BUDGET_EXHAUSTED:     a computation hit its resource budget.
- FORMAT_CORRUPT:       an input payload/archive is structurally broken.
- SEMANTIC_UNSOLVABLE:  input is well-formed but cannot yield an answer.
"""

OK = "ok"
BUDGET_EXHAUSTED = "budget_exhausted"
FORMAT_CORRUPT = "format_corrupt"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"
RECOVERED_INCOMPLETE = "recovered_incomplete"


def diag(path, code, message, severity="error"):
    """A located diagnostic. `path` points at the offending position."""
    return {"path": path, "code": code, "message": message, "severity": severity}


def make_result(status=OK, value=None, diagnostics=None):
    return {"status": status, "value": value, "diagnostics": list(diagnostics or [])}
