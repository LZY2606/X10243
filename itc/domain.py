"""Shared primitives: stable identity, digests, and the failure taxonomy.

Three distinct failure kinds are used everywhere in the system:
  BUDGET_EXHAUSTED     - a resource budget (evaluations / samples) ran out
  FORMAT_CORRUPT       - input bytes/structure could not be parsed
  SEMANTIC_UNSOLVABLE  - input parsed fine but cannot yield a unique answer
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
FORMAT_CORRUPT = "FORMAT_CORRUPT"
SEMANTIC_UNSOLVABLE = "SEMANTIC_UNSOLVABLE"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def make_id(prefix: str, obj) -> str:
    """Content-derived, deterministic identity: equivalent inputs -> same id."""
    return f"{prefix}-{digest(obj)[:16]}"


@dataclass
class Failure:
    kind: str  # one of BUDGET_EXHAUSTED / FORMAT_CORRUPT / SEMANTIC_UNSOLVABLE
    message: str
    diagnostics: list = field(default_factory=list)

    def to_dict(self):
        return {"failure": self.kind, "message": self.message, "diagnostics": self.diagnostics}


class BudgetExhausted(Exception):
    """Raised inside numeric kernels; converted to Failure at boundaries."""

    def __init__(self, message: str, used: int, budget: int):
        super().__init__(message)
        self.used = used
        self.budget = budget
