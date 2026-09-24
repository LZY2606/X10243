class ITCError(Exception):
    code = "itc_error"

    def __init__(self, message, *, path=None, details=None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}

    def to_diagnostic(self):
        data = {"code": self.code, "message": self.message, "details": self.details}
        if self.path is not None:
            data["path"] = self.path
        return data


class FormatCorrupted(ITCError):
    code = "format_corrupted"


class SemanticNoSolution(ITCError):
    code = "semantic_no_solution"


class ResourceBudgetExhausted(ITCError):
    code = "resource_budget_exhausted"


class ConflictRejected(ITCError):
    code = "conflict_rejected"

