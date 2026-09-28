"""三类必须可区分的失败结果，外加带位置的诊断。

- FormatCorrupt:   格式损坏（无法解析输入结构）
- SemanticInvalid: 语义无解（结构可解析，但领域约束不满足）
- BudgetExhausted: 资源预算耗尽（迭代/求值次数达到上限）

输入不完整时一律返回带位置的诊断，绝不用空值或默认值伪装成功。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    location: str = ""
    severity: str = "error"  # error | warning
    value: Any = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
            "value": self.value,
        }


class ITCError(Exception):
    kind = "itc_error"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupt(ITCError):
    """输入结构损坏：词法/语法层无法解析。"""

    kind = "format_corrupt"


class SemanticInvalid(ITCError):
    """语义无解：结构合法，但领域约束矛盾或信息不足。"""

    kind = "semantic_invalid"


class BudgetExhausted(ITCError):
    """资源预算耗尽：可计算但达到迭代/求值上限，结果不可信。"""

    kind = "budget_exhausted"

    def __init__(self, message: str, budget: int, used: int,
                 diagnostics: list[Diagnostic] | None = None):
        super().__init__(message, diagnostics)
        self.budget = budget
        self.used = used

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["budget"] = self.budget
        d["used"] = self.used
        return d
