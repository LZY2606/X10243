"""三类必须可区分的失败：资源预算耗尽、格式损坏、语义无解。

诊断（Diagnostic）一律带位置（材料/列/行/坐标），不使用空值或默认值伪装成功。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str  # "error" | "warning"
    message: str
    location: str = ""
    raw: Any = None
    rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diags_to_dict(items: list[Diagnostic]) -> list[dict[str, Any]]:
    return [d.to_dict() for d in items]


def dict_to_diags(items: list[dict[str, Any]]) -> list[Diagnostic]:
    return [Diagnostic(**d) for d in items]


class ITCError(Exception):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorruptError(ITCError):
    """格式损坏：无法解析的字节/结构，无法定位到合法材料。"""


class SemanticUnresolvableError(ITCError):
    """语义无解：结构可读，但领域语义冲突/无法唯一决定。"""


class ResourceBudgetExhaustedError(ITCError):
    """资源预算耗尽：例如拟合评估次数超限。"""
