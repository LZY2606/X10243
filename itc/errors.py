"""三类互不相同的失败结果：格式损坏 / 语义无解 / 资源预算耗尽。

所有领域错误都携带带位置的诊断；不得用空值或默认值伪装成功。
"""
from __future__ import annotations

from typing import Any


class ITCError(Exception):
    code = "itc_error"

    def __init__(self, message: str, *, diagnostics: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.diagnostics: list[dict[str, Any]] = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "diagnostics": self.diagnostics}


class FormatCorrupt(ITCError):
    """格式损坏：字节/结构无法解析，无法定位到任何领域对象。"""

    code = "format_corrupt"


class SemanticUnresolvable(ITCError):
    """语义无解：结构可解析，但领域约束不满足（如量纲不匹配、材料缺失）。"""

    code = "semantic_unresolvable"


class BudgetExhausted(ITCError):
    """资源预算耗尽：合法输入但超过允许的求值/迭代/体积预算。"""

    code = "budget_exhausted"


def diagnostic(
    path: str,
    message: str,
    *,
    severity: str = "error",
    raw: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    diag: dict[str, Any] = {"severity": severity, "path": path, "message": message}
    if raw is not None:
        diag["raw"] = raw
    diag.update(extra)
    return diag
