"""三种必须互相区分的失败结果。

- FORMAT_CORRUPTED      格式损坏：字节/结构层面无法解析
- SEMANTIC_UNRESOLVABLE 语义无解：结构合法但业务上无法唯一决定或缺必需输入
- BUDGET_EXHAUSTED      资源预算耗尽：在固定预算内无法完成（行数/求值次数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    path: str
    message: str
    value: Any = None

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message, "value": self.value}


class ITCError(Exception):
    code = "ITC_ERROR"
    http_status = 400

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupted(ITCError):
    code = "FORMAT_CORRUPTED"
    http_status = 422


class SemanticUnresolvable(ITCError):
    code = "SEMANTIC_UNRESOLVABLE"
    http_status = 422


class BudgetExhausted(ITCError):
    code = "BUDGET_EXHAUSTED"
    http_status = 507
