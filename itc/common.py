"""Shared diagnostics and error types."""


def diagnostic(code, message, location):
    """A located diagnostic. Diagnostics are data, not log lines: they are
    stored on materials / computations and exported with the archive."""
    return {"code": code, "message": message, "location": location}


class ItcError(Exception):
    """Structured failure. ``status`` is one of:
    - format_corrupt:      the input cannot be parsed at all
    - budget_exhausted:    a computational budget ran out
    - semantic_unsolvable: the input parses but admits no solution
    These three outcomes are always distinct results, never collapsed."""

    def __init__(self, status, diagnostics):
        self.status = status
        self.diagnostics = diagnostics if isinstance(diagnostics, list) else [diagnostics]
        super().__init__(f"{status}: {self.diagnostics[0].get('message') if self.diagnostics else ''}")
