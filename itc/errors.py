class ITCError(Exception):
    code = "itc_error"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorrupt(ITCError):
    code = "format_corrupt"


class SemanticImpossible(ITCError):
    code = "semantic_impossible"


class BudgetExhausted(ITCError):
    code = "budget_exhausted"


def diagnostic(path, message, value=None, code="invalid_value"):
    item = {"path": path, "code": code, "message": message}
    if value is not None:
        item["value"] = value
    return item
