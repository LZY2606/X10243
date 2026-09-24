"""三种必须可区分的失败结果：资源预算耗尽 / 格式损坏 / 语义无解。"""

STATUS_OK = "ok"
STATUS_BUDGET = "budget_exhausted"
STATUS_FORMAT = "format_corrupt"
STATUS_SEMANTIC = "semantic_unsolvable"


class BudgetExhausted(Exception):
    """计算预算（如模型评估次数）耗尽。"""
    status = STATUS_BUDGET


class FormatCorrupt(Exception):
    """输入字节流 / 结构损坏，无法解析。"""
    status = STATUS_FORMAT


class SemanticUnsolvable(Exception):
    """格式合法但语义上无法唯一求解或输入不完整。"""
    status = STATUS_SEMANTIC
