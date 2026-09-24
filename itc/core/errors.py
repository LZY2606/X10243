"""三类互不等价的失败结果与带位置的诊断。

- format_corrupt：字节/词法/结构层面无法解析（坏 CSV、缺列、截断）。
- semantic_unsolvable：可解析但领域语义无解（自相矛盾的时间/端点、
  浓度为负、无法唯一决定且无合法候选等）。
- resource_budget_exhausted：输入本身可解，但迭代/时间/规模预算耗尽。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import ERR_BUDGET, ERR_FORMAT, ERR_SEMANTIC


@dataclass(frozen=True)
class Location:
    """诊断位置：材料槽 / 注射序号 / 行 / 字段 / 时间坐标。"""

    slot: str | None = None
    injection: str | int | None = None
    row: int | None = None
    field: str | None = None
    at: Any = None

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v is not None}
        return out


class DomainError(Exception):
    kind = "domain_error"

    def __init__(self, message: str, location: Location | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "message": self.message}
        if self.location is not None:
            out["location"] = self.location.to_dict()
        if self.details:
            out["details"] = self.details
        return out


class FormatCorruptError(DomainError):
    kind = ERR_FORMAT


class SemanticUnsolvableError(DomainError):
    kind = ERR_SEMANTIC


class ResourceBudgetExhaustedError(DomainError):
    kind = ERR_BUDGET


@dataclass
class Diagnostic:
    """非致命诊断：值留在原位置，诊断随行。"""

    code: str
    message: str
    severity: str = "error"
    location: Location | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"code": self.code, "message": self.message,
               "severity": self.severity}
        if self.location is not None:
            out["location"] = self.location.to_dict()
        return out


def diag_list(diags: list[Diagnostic]) -> list[dict[str, Any]]:
    return [d.to_dict() for d in diags]


@dataclass
class IncompleteInput(Exception):
    """输入不完整：携带位置，禁止以空值/默认值伪装成功。"""

    message: str
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": ERR_SEMANTIC,
            "message": self.message,
            "diagnostics": diag_list(self.diagnostics),
        }
