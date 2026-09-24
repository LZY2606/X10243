"""Positioned diagnostics and the three distinct terminal failure classes.

The pipeline must never report success using nulls or defaults. Three classes
of terminal failure are deliberately separate:

* ``BUDGET_EXHAUSTED`` - resource budget exhausted (iteration / time limit)
* ``FORMAT_CORRUPT``    - the input representation is structurally unusable
* ``SEMANTIC_UNRESOLVED`` - the input is well formed but has no unique/legal
  interpretation under the stated rules
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
FORMAT_CORRUPT = "FORMAT_CORRUPT"
SEMANTIC_UNRESOLVED = "SEMANTIC_UNRESOLVED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: str = Severity.ERROR.value
    # human/script locatable position: file, row, column, injection index, ...
    location: dict = field(default_factory=dict)
    # the original representation that triggered the diagnostic, if any
    raw: object | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def diagnostic(code: str, message: str, *, severity: str = Severity.ERROR.value,
               location: dict | None = None, raw: object | None = None) -> Diagnostic:
    return Diagnostic(code, message, severity, location or {}, raw)


def collect_dicts(diags: list[Diagnostic]) -> list[dict]:
    return [d.to_dict() for d in diags]
