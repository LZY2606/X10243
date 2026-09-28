"""三种语义不同的失败结果。

* FormatCorruptError   —— 文件结构损坏，无法解析（格式层）。
* BudgetExhaustedError —— 解析/计算资源预算耗尽（防护层，区别于格式损坏）。
* SemanticError        —— 结构可解析但领域语义无解（如浓度单位无法转换）。

三者都是带位置的诊断集合的载体，绝不以空值/默认值伪装成功。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Diagnostic:
    code: str
    message: str
    location: str = ""
    severity: str = "error"

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "location": self.location, "severity": self.severity}


class ITCError(Exception):
    def __init__(self, message: str, diagnostics: List[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: List[Diagnostic] = diagnostics or []

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": str(self),
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }

    kind = "itc_error"


class FormatCorruptError(ITCError):
    kind = "format_corrupt"


class BudgetExhaustedError(ITCError):
    kind = "budget_exhausted"


class SemanticError(ITCError):
    kind = "semantic_unsolvable"
