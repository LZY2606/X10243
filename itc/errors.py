"""Three distinct failure outcomes mandated by the spec.

* FormatCorrupt     - the byte/CSV structure is unreadable / damaged
* SemanticImpasse   - structure parses but the data cannot mean what it must
* BudgetExhausted   - the evaluatable input is valid but the resource budget ran out

Parse-level "unconvertible value" situations are NOT failures: the original
representation is retained in place and a diagnostic is attached (units.py /
importer.py).  They surface as ``diagnostics`` lists, never as exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ITCError(Exception):
    kind = "error"


class FormatCorrupt(ITCError):
    """Structural damage: cannot even recover typed rows."""

    kind = "format_corrupt"


class SemanticImpasse(ITCError):
    """The structure is fine, but the content admits no valid interpretation."""

    kind = "semantic_impasse"


class BudgetExhausted(ITCError):
    """Input was valid and evaluatable, but the resource budget was spent."""

    kind = "budget_exhausted"


@dataclass
class Diagnostic:
    """A non-fatal, position-bearing problem attached to imported material."""

    code: str
    path: str
    message: str
    raw: Any = None
    at: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "raw": self.raw,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Diagnostic":
        return cls(
            code=d["code"],
            path=d["path"],
            message=d["message"],
            raw=d.get("raw"),
            at=d.get("at", {}),
        )
