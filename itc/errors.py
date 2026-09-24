"""Typed failure results.

资源预算耗尽、格式损坏、语义无解是三种互不相同的结果，
分别用三个异常类型表达，调用方（含测试）可直接区分。
"""


class ITCError(Exception):
    code = "itc_error"

    def __init__(self, message, *, locations=None):
        super().__init__(message)
        self.locations = list(locations or [])


class FormatCorrupt(ITCError):
    """文件无法解析：不是“不完整”，而是已损坏。"""

    code = "format_corrupt"


class SemanticImpossible(ITCError):
    """语法合法但领域语义无解（单位不可换算、量缺失、时间基准冲突等）。"""

    code = "semantic_impossible"


class ResourceExhausted(ITCError):
    """计算超过显式预算（网格规模、迭代次数、样本点数）。"""

    code = "resource_exhausted"
