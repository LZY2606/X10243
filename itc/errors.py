"""三类必须互相区分的结果：资源预算耗尽、格式损坏、语义无解。"""


class ItcError(Exception):
    kind = "ITC_ERROR"

    def __init__(self, message, *, code=None, path=None, details=None):
        super().__init__(message)
        self.message = message
        self.code = code or self.kind
        self.path = path
        self.details = details or {}

    def to_dict(self):
        return {
            "kind": self.kind,
            "code": self.code,
            "message": str(self.message),
            "path": self.path,
            "details": self.details,
        }


class CorruptFormat(ItcError):
    """格式损坏：词法/结构层面无法解析（坏 CSV、坏 zip、截断 JSON 等）。"""

    kind = "CORRUPT_FORMAT"


class SemanticNoSolution(ItcError):
    """语义无解：结构合法，但领域上没有任何合法结果（没有可用积分点等）。"""

    kind = "SEMANTIC_NO_SOLUTION"


class BudgetExhausted(ItcError):
    """资源预算耗尽：合法但在允许的模型评估次数内未能完成。"""

    kind = "BUDGET_EXHAUSTED"


class Diagnostic:
    """非致命诊断：值留在原位置，不伪装成成功。"""

    __slots__ = ("code", "severity", "path", "message", "related")

    def __init__(self, code, message, *, severity="warning", path=None, related=None):
        self.code = code
        self.severity = severity
        self.path = path
        self.message = message
        self.related = related or []

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
            "related": list(self.related),
        }
