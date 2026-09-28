"""结构化结果码与带位置的诊断。

三种必须可区分的非成功结果：
- FORMAT_CORRUPTION  格式损坏（无法解析/校验失败，附字节或记录位置）
- SEMANTIC_UNSOLVABLE 语义无解（规则上无法决定，例如空白对应缺失）
- BUDGET_EXHAUSTED   资源预算耗尽（拟合迭代/计算预算用尽，尚未收敛）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


class Outcome:
    OK = "ok"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FORMAT_CORRUPTION = "format_corruption"
    SEMANTIC_UNSOLVABLE = "semantic_unsolvable"


class DiagnosticCode:
    # 导入层
    UNCONVERTIBLE_VALUE = "unconvertible_value"
    MISSING_FIELD = "missing_field"
    FORMAT_CORRUPTION = "format_corruption"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    MISSING_REFERENCE = "missing_reference"
    # 积分层
    GAP_CROSSES_WINDOW = "gap_crosses_window"
    WINDOW_BEYOND_DATA = "window_beyond_data"
    OVERLAP_PEAKS = "overlap_peaks"
    OVERLAP_UNRESOLVED = "overlap_unresolved"
    FIRST_INJECTION_DISCARDED = "first_injection_discarded"
    INVALID_SAMPLE = "invalid_sample"
    # 稀释热层
    BLANK_CORRESPONDENCE_MISSING = "blank_correspondence_missing"
    BLANK_CORRESPONDENCE_RULE_UNSTATED = "blank_correspondence_rule_unstated"
    # 模型层
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODELS_EQUALLY_GOOD = "models_equally_good"
    MODEL_EXPIRED = "model_expired"
    FIT_ILL_CONDITIONED = "fit_ill_conditioned"
    INPUT_INCOMPLETE = "input_incomplete"
    # 决定层
    CONFLICTING_DECISIONS = "conflicting_decisions"


@dataclass
class Diagnostic:
    """机器可读取的诊断；位置必须指向具体材料版本/注射序号/时间区间。"""

    code: str
    message: str
    outcome: str = Outcome.OK
    positions: list[dict[str, Any]] = field(default_factory=list)
    affected: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Diagnostic":
        return cls(**data)


class ITCError(Exception):
    """携带结构化诊断的领域错误。"""

    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def raise_format(message: str, *, position: dict[str, Any] | None = None, **kw) -> None:
    positions = [position] if position else []
    raise ITCError(
        Diagnostic(DiagnosticCode.FORMAT_CORRUPTION, message,
                   Outcome.FORMAT_CORRUPTION, positions, **kw)
    )
