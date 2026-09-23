"""Diagnostics and failure kinds for the ITC review pipeline.

Three failure outcomes are deliberately distinct:
  - BUDGET_EXHAUSTED: a caller-supplied resource budget ran out.
  - FORMAT_CORRUPT: bytes/JSON could not be parsed or the DB is torn.
  - SEMANTIC_UNSOLVABLE: input parsed fine but references/constraints cannot be resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


BUDGET_EXHAUSTED = "budget_exhausted"
FORMAT_CORRUPT = "format_corrupt"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = ""           # machine-readable position, e.g. "run.injections[7].window"
    severity: str = "error"  # error | warning | info

    def to_dict(self) -> dict:
        return asdict(self)


class ItcFailure(Exception):
    """A structured failure; `kind` is one of the three outcome kinds above."""

    def __init__(self, kind: str, message: str, diagnostics: Optional[list] = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.diagnostics: list[Diagnostic] = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }
