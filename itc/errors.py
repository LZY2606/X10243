class DomainError(Exception):
    kind = "semantic_no_solution"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []


class CorruptFormatError(DomainError):
    kind = "corrupt_format"


class SemanticNoSolutionError(DomainError):
    kind = "semantic_no_solution"


class ResourceBudgetExceededError(DomainError):
    kind = "resource_budget_exhausted"


def diagnostic(path, message, value=None, valid_range=None):
    item = {"path": path, "message": message}
    if value is not None:
        item["actual"] = value
    if valid_range is not None:
        item["valid_range"] = valid_range
    return item
