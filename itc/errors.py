from dataclasses import dataclass, field


class ITCError(Exception):
    code = "itc_error"

    def __init__(self, message, path=None, details=None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }


class ResourceBudgetExhausted(ITCError):
    code = "resource_budget_exhausted"


class FormatCorrupt(ITCError):
    code = "format_corrupt"


class SemanticUnresolvable(ITCError):
    code = "semantic_unresolvable"


@dataclass
class Diagnostic:
    severity: str
    code: str
    message: str
    path: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }
