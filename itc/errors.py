"""三种彼此不同的失败结果（任务硬性要求）。

- FormatCorrupt        资源/输入本身结构损坏，无法解析
- SemanticInvalid      结构可解析但领域语义无解 / 输入不完整（带位置诊断）
- BudgetExhausted      格式与语义均合法，但资源预算耗尽
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    """带位置的诊断；位置指向原始表示中的材料/字段/索引。"""

    code: str
    message: str
    location: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "value": self.value,
        }


class ITCError(Exception):
    kind = "itc_error"


@dataclass
class FormatCorrupt(ITCError):
    """格式损坏：字节/结构无法解析。与语义问题严格区分。"""

    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)


@dataclass
class SemanticInvalid(ITCError):
    """语义无解：解析成功但领域上不合法/不完整。必须携带带位置诊断。"""

    message: str
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "semantic_invalid",
            "message": self.message,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass
class BudgetExhausted(ITCError):
    """资源预算耗尽：数据合法，但计算在预算内无法完成。"""

    message: str
    budget: int
    used: int

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "budget_exhausted",
            "message": self.message,
            "budget": self.budget,
            "used": self.used,
        }
