"""Typed failure outcomes.

The specification requires three *distinct* results rather than one generic error:

* :class:`BudgetExhausted`  - a computation spent its bounded resource budget
  (iterations / candidates / points) without reaching a solution.
* :class:`CorruptFormat`    - bytes/text could not even be parsed against the
  declared format; the offending location is reported.
* :class:`SemanticImpasse`  - the data parsed but has no unique/legal meaning
  (missing blank counterpart, window crosses a gap, unconvertible unit, ...).
"""

from __future__ import annotations

from typing import Any


class ITCError(Exception):
    """Base class carrying a stable ``code`` and optional structured ``context``."""

    code = "itc_error"

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class BudgetExhausted(ITCError):
    """Resource budget (iteration / candidate / point) was exhausted."""

    code = "budget_exhausted"

    def __init__(self, message: str, *, budget: int | None = None, used: int | None = None,
                 context: dict[str, Any] | None = None):
        ctx = dict(context or {})
        if budget is not None:
            ctx["budget"] = budget
        if used is not None:
            ctx["used"] = used
        super().__init__(message, context=ctx)


class CorruptFormat(ITCError):
    """The input does not conform to its declared on-disk format."""

    code = "corrupt_format"

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None,
                 column: int | None = None, context: dict[str, Any] | None = None):
        ctx = dict(context or {})
        if path is not None:
            ctx["path"] = path
        if line is not None:
            ctx["line"] = line
        if column is not None:
            ctx["column"] = column
        super().__init__(message, context=ctx)


class SemanticImpasse(ITCError):
    """The input parsed but is incomplete / ambiguous / has no legal interpretation."""

    code = "semantic_impasse"

    def __init__(self, message: str, *, locations: list[Any] | None = None,
                 context: dict[str, Any] | None = None):
        ctx = dict(context or {})
        if locations:
            ctx["locations"] = list(locations)
        super().__init__(message, context=ctx)


def diagnostic(code: str, message: str, **locations: Any) -> dict[str, Any]:
    """A non-fatal diagnostic record attached to the object at its original place.

    Unconvertible / incomplete values stay in place (never dropped, never replaced
    by a default); they carry one of these diagnostics.
    """
    loc = {k: v for k, v in locations.items() if v is not None}
    return {"code": code, "message": message, "locations": loc}
