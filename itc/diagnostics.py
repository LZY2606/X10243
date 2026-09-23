"""Structured diagnostics and result statuses.

Three failure outcomes are kept strictly distinct:
  - BUDGET_EXHAUSTED:   a resource budget (e.g. max model evaluations) ran out.
  - FORMAT_CORRUPT:     bytes/JSON could not be parsed or failed schema checks.
  - SEMANTIC_UNSOLVABLE: input parsed fine but is semantically insufficient
                         (unknown units, no usable injections, ...).
A successful computation may still carry non-fatal diagnostics (status OK or
DIAGNOSTIC); inputs are never silently replaced by defaults.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


class Status:
    OK = "ok"
    DIAGNOSTIC = "diagnostic"                # computed, but with warnings
    STALE = "stale"                          # superseded by a newer dependency version
    BUDGET_EXHAUSTED = "budget_exhausted"
    FORMAT_CORRUPT = "format_corrupt"
    SEMANTIC_UNSOLVABLE = "semantic_unsolvable"


FAILURE_STATUSES = (Status.BUDGET_EXHAUSTED, Status.FORMAT_CORRUPT, Status.SEMANTIC_UNSOLVABLE)


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = ""          # machine-addressable position, e.g. "injections[3].volume"
    severity: str = "error"  # "error" | "warning"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Outcome:
    """Result of a computation: status + payload + diagnostics."""
    status: str
    payload: Any = None
    diagnostics: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (Status.OK, Status.DIAGNOSTIC, Status.STALE)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "payload": self.payload,
            "diagnostics": [d.to_dict() if isinstance(d, Diagnostic) else d for d in self.diagnostics],
        }

    def digest_payload(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)
