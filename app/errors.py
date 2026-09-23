"""Distinct error outcomes mandated by the domain:

* FORMAT_CORRUPTED    - the material cannot be parsed structurally.
* SEMANTIC_NO_SOLUTION - parseable, but no unique/legal domain answer exists.
* BUDGET_EXHAUSTED    - a declared resource budget ran out before completion.
"""


class ItcError(Exception):
    code = "ITC_ERROR"

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = list(diagnostics or [])

    def to_dict(self):
        return {"code": self.code, "message": self.message, "diagnostics": self.diagnostics}


class FormatCorrupted(ItcError):
    code = "FORMAT_CORRUPTED"

    def __init__(self, message, *, path=None, line=None, column=None, diagnostics=None):
        super().__init__(message, diagnostics=diagnostics)
        self.path = path
        self.line = line
        self.column = column

    def to_dict(self):
        data = super().to_dict()
        if self.path is not None:
            data["path"] = self.path
        if self.line is not None:
            data["line"] = self.line
        if self.column is not None:
            data["column"] = self.column
        return data


class SemanticNoSolution(ItcError):
    code = "SEMANTIC_NO_SOLUTION"


class BudgetExhausted(ItcError):
    code = "BUDGET_EXHAUSTED"

    def __init__(self, message, *, budget=None, used=None, partial=None, diagnostics=None):
        super().__init__(message, diagnostics=diagnostics)
        self.budget = budget
        self.used = used
        self.partial = partial

    def to_dict(self):
        data = super().to_dict()
        data["budget"] = self.budget
        data["used"] = self.used
        data["partial"] = self.partial
        return data
