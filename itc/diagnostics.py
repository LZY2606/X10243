"""Structured diagnostics.

Three failure categories are deliberately distinct machine-readable results:
  - format_corrupt:        the input bytes/text cannot be parsed at all
  - semantic_unsolvable:   parsed fine, but the request cannot be decided
  - budget_exhausted:      a computation ran out of its resource budget
Data-quality and scope notes never fail a computation by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

FORMAT_CORRUPT = "format_corrupt"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"
BUDGET_EXHAUSTED = "budget_exhausted"
DATA_QUALITY = "data_quality"
SCOPE_NOTE = "scope_note"

FAILURE_CATEGORIES = (FORMAT_CORRUPT, SEMANTIC_UNSOLVABLE, BUDGET_EXHAUSTED)


@dataclass
class Diagnostic:
    code: str
    category: str
    location: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Diagnostic":
        return Diagnostic(code=d["code"], category=d["category"],
                          location=d["location"], message=d["message"])


@dataclass
class Outcome:
    """Result of an operation. `status` is one of:
    ok | partial | format_corrupt | semantic_unsolvable | budget_exhausted
    A non-ok failure status never carries a fabricated value."""
    status: str
    value: Optional[Any] = None
    diagnostics: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "value": self.value,
            "diagnostics": [d.to_dict() if isinstance(d, Diagnostic) else d
                            for d in self.diagnostics],
        }


def fail(category: str, code: str, location: str, message: str) -> Outcome:
    assert category in FAILURE_CATEGORIES
    return Outcome(status=category, value=None,
                   diagnostics=[Diagnostic(code, category, location, message)])
