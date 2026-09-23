"""带位置的诊断，以及三种必须彼此可区分的失败结果。

- FormatCorrupt      格式损坏：词法/结构层面无法解析
- SemanticUnresolvable 语义无解：结构合法但领域含义无法成立
- BudgetExhausted    资源预算耗尽：输入合法但给定预算内无法完成
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    path: str = ""
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ITCError(Exception):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorrupt(ITCError):
    kind = "format_corrupt"


class SemanticUnresolvable(ITCError):
    kind = "semantic_unresolvable"


class BudgetExhausted(ITCError):
    kind = "budget_exhausted"


def diag_payload(exc: ITCError) -> dict[str, Any]:
    return {
        "error_kind": exc.kind,
        "message": str(exc),
        "diagnostics": [d.as_dict() for d in exc.diagnostics],
    }
