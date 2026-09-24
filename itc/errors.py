"""Three distinct failure outcomes required by the specification.

* :class:`FormatCorrupt`   - the bytes/structure cannot be parsed at all.
* :class:`SemanticImpasse` - parsed, but the request has no legal meaning
                             (e.g. a unit that cannot be converted, a window
                             that crosses a data gap, an unsolvable model).
* :class:`BudgetExhausted` - the meaning is clear and the data is valid but
                             the configured evaluation budget ran out.

Every failure carries positional diagnostics (file / row / column / time
coordinate) so callers never have to fake success with nulls or defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    file: str | None = None
    row: int | None = None          # 1-based row in the source representation
    column: str | None = None       # column name when known
    t_s: float | None = None        # time coordinate (seconds) when applicable
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {}}


class ITCError(Exception):
    kind = "itc_error"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: list[Diagnostic] = diagnostics or []

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": self.kind,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupt(ITCError):
    """The representation itself is damaged (bad CSV, unknown framing, ...)."""

    kind = "format_corrupt"


class SemanticImpasse(ITCError):
    """Parsable input whose requested interpretation has no legal answer."""

    kind = "semantic_impasse"


class BudgetExhausted(ITCError):
    """Valid input, clear intent, but the resource budget was consumed."""

    kind = "budget_exhausted"

    def __init__(self, message: str, used: int, limit: int,
                 diagnostics: list[Diagnostic] | None = None):
        super().__init__(message, diagnostics)
        self.used = used
        self.limit = limit

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["used"] = self.used
        payload["limit"] = self.limit
        return payload
