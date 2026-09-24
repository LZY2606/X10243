"""三种互相独立的失败结果。

* FormatCorrupt   —— 输入表示无法解析（词法/结构层）
* SemanticUnresolved —— 输入结构合法但语义上无法唯一决定（业务层）
* BudgetExhausted —— 计算在资源预算（步数/样本数）内未收敛
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    """带位置的诊断。loc 使用 JSON-Pointer 风格路径。"""

    code: str
    message: str
    loc: str = ""
    value: object | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "loc": self.loc, "value": self.value}


class ItcError(Exception):
    outcome = "error"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: list[Diagnostic] = diagnostics or []

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupt(ItcError):
    outcome = "format_corrupt"


class SemanticUnresolved(ItcError):
    outcome = "semantic_unresolved"


class BudgetExhausted(ItcError):
    outcome = "budget_exhausted"


@dataclass
class Outcome:
    """服务层统一返回：status ∈ ok | format_corrupt | semantic_unresolved | budget_exhausted。"""

    status: str
    data: dict | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @staticmethod
    def ok(data: dict) -> "Outcome":
        return Outcome("ok", data, [])

    @staticmethod
    def fail(exc: ItcError) -> "Outcome":
        return Outcome(exc.outcome, None, exc.diagnostics)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "data": self.data,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"
