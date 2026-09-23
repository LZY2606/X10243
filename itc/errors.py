"""三类必须互相区分的失败结果，以及带位置的诊断。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---- 结构化诊断 ---------------------------------------------------------

@dataclass
class Diagnostic:
    code: str            # 稳定诊断码，如 unit_unknown / gap_clipped / blank_missing
    message: str         # 面向审阅者的说明
    path: str = ""       # 位置，如 "power.rows[12]"、"injections[3].time"
    severity: str = "error"  # error | warning | info

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- 三类不同失败 -------------------------------------------------------

class ITCError(Exception):
    """业务层基类，绝不作为吞错的出口。"""


class FormatCorrupt(ITCError):
    """格式损坏：字节/语法层面无法解析。"""


class SemanticUnsolvable(ITCError):
    """语义无解：能解析，但领域含义无法唯一决定或条件不足。"""


class BudgetExhausted(ITCError):
    """资源预算耗尽：迭代/操作次数达到上限。"""


STATUS_OK = "ok"
STATUS_FORMAT_CORRUPT = "format_corrupt"
STATUS_SEMANTIC_UNSOLVABLE = "semantic_unsolvable"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class Outcome:
    """所有业务操作的统一返回：状态互斥，绝不返回伪装成功。"""
    status: str
    value: Any = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @classmethod
    def good(cls, value: Any, diagnostics: list[Diagnostic] | None = None) -> "Outcome":
        return cls(STATUS_OK, value, diagnostics or [])

    @classmethod
    def fail(cls, status: str, diagnostics: list[Diagnostic]) -> "Outcome":
        return cls(status, None, diagnostics)


# ---- 资源预算 -----------------------------------------------------------

class Budget:
    """显式资源预算；耗尽抛 BudgetExhausted，与格式/语义失败区分。"""

    def __init__(self, max_ops: int = 100_000, label: str = "op"):
        self.max_ops = max_ops
        self.remaining = max_ops
        self.label = label

    def spend(self, n: int = 1, where: str = "") -> None:
        self.remaining -= n
        if self.remaining < 0:
            raise BudgetExhausted(
                f"budget exhausted at {self.label}:{where} "
                f"(max_ops={self.max_ops})"
            )
