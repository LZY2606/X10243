"""Three distinct failure results: budget, format, semantic."""


class ItcError(Exception):
    """Base error. `kind` is machine-readable: budget | format | semantic."""

    kind = "semantic"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []

    def to_dict(self):
        return {"kind": self.kind, "message": self.message, "diagnostics": self.diagnostics}


class BudgetExhausted(ItcError):
    """Resource budget (evaluation/step cap) ran out."""

    kind = "budget"


class FormatCorrupt(ItcError):
    """Input bytes/structure are damaged and cannot be parsed."""

    kind = "format"


class SemanticUnsolvable(ItcError):
    """Input parses but references/constraints cannot be resolved."""

    kind = "semantic"
