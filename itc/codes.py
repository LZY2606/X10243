"""稳定编码：材料类型、状态、诊断代码、失败分类、模型身份。

这些字符串会写入事件与导出档案，因此保持稳定，不随显示文案改动。
"""
from __future__ import annotations

import enum


class FailureKind(str, enum.Enum):
    """资源预算耗尽、格式损坏、语义无解是三种不同结果。"""
    BUDGET_EXHAUSTED = "budget_exhausted"
    FORMAT_CORRUPT = "format_corrupt"
    SEMANTIC = "semantic_unresolvable"


class MaterialKind(str, enum.Enum):
    POWER = "power"                 # 功率采样（含点流）
    INJECTIONS = "injections"       # 注射体积和时间
    CONCENTRATION = "concentration"  # 浓度（池/注射器，逐注）
    TEMPERATURE = "temperature"
    CELL_VOLUME = "cell_volume"
    BASELINE = "baseline"           # 样品基线曲线（单独版本化）
    BLANK_POWER = "blank_power"     # 空白功率曲线（单独版本化）
    BLANK_INJECTIONS = "blank_injections"
    BLANK_BASELINE = "blank_baseline"
    BLANK_RULE = "blank_rule"       # 稀释热对应规则（不可变决定材料）


class ResultType(str, enum.Enum):
    INTEGRATION = "integration"
    FIT = "fit"


class WinStatus(str, enum.Enum):
    OK = "ok"
    DISCARDED = "discarded_first"
    GAP_BLOCKED = "gap_blocked"
    OVERLAP = "overlap"
    OVERLAP_GAP_BLOCKED = "overlap_gap_blocked"


class FitStatus(str, enum.Enum):
    OK = "ok"
    EQUALLY_OPTIMAL = "equally_optimal"
    UNIDENTIFIABLE = "unidentifiable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INSUFFICIENT_POINTS = "insufficient_points"


class ModelId(str, enum.Enum):
    ONE_SITE = "one_site"
    TWO_SITE = "two_site"
    DILUTION = "dilution"


class Decision(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# 诊断代码（稳定身份，测试直接断言）
DIAG = {
    "FIRST_DISCARD": "first_injection_discarded",
    "OVERLAP_PEAK": "overlapping_post_injection_peak",
    "GAP_SPAN": "integration_window_spans_data_gap",
    "ENDPOINT_PAST_END": "endpoint_past_trace_end",
    "ENDPOINT_ORDER": "manual_endpoint_before_start",
    "ENDPOINT_ORDER": "manual_endpoint_before_start",
    "ENDPOINT_COLLISION": "manual_endpoint_past_next_injection",
    "BASELINE_EXTRAPOLATED": "baseline_extrapolated_outside_anchors",
    "BLANK_UNMATCHED": "blank_injection_correspondence_missing",
    "BLANK_RULE_UNDECLARED": "blank_rule_correspondence_undeclared",
    "CONC_REVISED": "concentration_revision_supersedes_fit",
    "LOCAL_CHANGE": "local_source_change_invalidates_window",
    "UNCONVERTIBLE_POINT": "value_unconvertible_left_in_place",
    "UNCONVERTIBLE_SCALAR": "scalar_unconvertible_left_in_place",
    "EQUAL_OPTIMA": "multiple_equally_optimal_models",
    "CONFLICTING_DECISION": "conflicting_manual_decisions",
}

# 积分结果状态是否产热
PRODUCES_HEAT = {
    WinStatus.OK, WinStatus.OVERLAP,
}
# AIC 等优容差（ΔAIC <= 该值视为等优，全部保留）
AIC_TIE_TOLERANCE = 2.0
