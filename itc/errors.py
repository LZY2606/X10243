"""Distinct failure outcomes required by the domain rules.

* ResourceExhausted  - budget (iteration/points) exhausted; a retry with a
  larger budget could still succeed.
* FormatCorrupt      - bytes/syntax are unusable before any semantics.
* SemanticImpossible - syntactically valid input that cannot mean what is
  claimed (e.g. integrating a window that falls outside every valid span).
"""

from __future__ import annotations


class ITCError(Exception):
    kind = "itc_error"

    def __init__(self, message: str, *, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "diagnostics": self.diagnostics}


class ResourceExhausted(ITCError):
    kind = "resource_exhausted"

    def __init__(self, message: str, *, budget: str, used: int, limit: int, diagnostics=None):
        super().__init__(message, diagnostics=diagnostics)
        self.budget = budget
        self.used = used
        self.limit = limit

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"budget": self.budget, "used": self.used, "limit": self.limit})
        return d


class FormatCorrupt(ITCError):
    kind = "format_corrupt"


class SemanticImpossible(ITCError):
    kind = "semantic_impossible"
