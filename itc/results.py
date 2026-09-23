"""Computation result taxonomy.

Three terminal failure states are deliberately distinct and must never be
collapsed into each other or into a fake success:
- budget_exhausted: a resource budget ran out (position of exhaustion reported)
- format_corrupt: input bytes/text could not be parsed or validated
- semantic_unsolvable: parsed fine, but the request is meaningless domain-wise
"""
from __future__ import annotations

from dataclasses import dataclass, field

OK = "ok"
DIAGNOSTICS = "diagnostics"
BUDGET_EXHAUSTED = "budget_exhausted"
FORMAT_CORRUPT = "format_corrupt"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"
TERMINAL_FAILURES = (BUDGET_EXHAUSTED, FORMAT_CORRUPT, SEMANTIC_UNSOLVABLE)


@dataclass
class Result:
    status: str
    value: object = None
    diagnostics: list = field(default_factory=list)

    @property
    def ok(self):
        return self.status in (OK, DIAGNOSTICS)

    def to_json(self):
        return {
            "status": self.status,
            "value": self.value,
            "diagnostics": [d.to_json() if hasattr(d, "to_json") else d for d in self.diagnostics],
        }
